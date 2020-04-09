import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.Multimap;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.joda.time.DateTimeZone;
import org.neuroph.core.NeuralNetwork;
import org.neuroph.core.data.DataSet;
import org.neuroph.core.data.DataSetRow;
import org.neuroph.core.events.LearningEvent;
import org.neuroph.core.events.LearningEventListener;
import org.neuroph.core.learning.error.MeanAbsoluteError;
import org.neuroph.core.learning.error.MeanSquaredError;
import org.neuroph.nnet.MultiLayerPerceptron;
import org.neuroph.nnet.learning.MomentumBackpropagation;
import org.neuroph.util.DataSetColumnType;
import org.neuroph.util.TransferFunctionType;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URL;
import java.util.*;

public class CasePredictor implements LearningEventListener {

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILE = "covid_19_counties";
    private static final String STATES_FILE = "covid_19_states";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.dir") + "/src/main", "resources");

    private static final double learningRate = 0.01;
    private static final int numInputs = 2;
    private static final int numOutputs = 1;

    private static Map<String, Integer> locationToFIPS = new HashMap<>();
    private static Multimap<Integer, List<Writable>> toWrite = ArrayListMultimap.create();
    private static File dir = new File(DATA_PATH);

    public static void main(String[] args) {
        (new CasePredictor()).buildMolel();

//        makePrediction("Snohomish, Washington", "2020-04-03");
//        makePrediction("Los Angeles, California", "2020-04-03");
//        makePrediction("Hawaii", "2020-04-03");
//        makePrediction("New York", "2020-04-03");

        while (true) {
            System.out.print("");
        }

        // TODO: GUI for makePrediction
    }

    private void buildMolel() {
        // train and test the model using county data
        buildModel(COUNTIES_FILE, COUNTRIES_URL);
        // train and test the model using state data
        buildModel(STATES_FILE, STATES_URL);
    }

    private void makePrediction(String location, String date) {
        // perhaps give a dropdown to select county,
        // and calendar excluding today and past
        // take in a county and a date --> predict cases);
        boolean inCountyMode = location.contains(",");
        File rawInput = new File(FilenameUtils.concat(dir.getPath(), "input.csv"));
        try (FileWriter writer = new FileWriter(rawInput)) {
            writer.write(date + "," + locationToFIPS.get(location) + ",0");
            File cleanInput = cleanData(rawInput, inCountyMode);
            DataSet input = DataSet.createFromFile(cleanInput.getPath(), numInputs, numOutputs, ",");
            input.setColumnNames(new String[]{"date", "fips", "cases"});
            input.setColumnType(1, DataSetColumnType.NOMINAL);
            input.setLabel("cases");

            NeuralNetwork model = getModel();
            model.setInput(input.getRowAt(0).getInput());
            model.calculate();
            double[] prediction = model.getOutput();
            System.out.println("I predict there will be " + prediction[0] + " cases in " + location + " on " + date + ".");
        } catch (Exception e) {
            e.printStackTrace();
        }

    }

    private void buildModel(String filename, String url) {
        boolean inCountyMode = filename.contains("counties");
        try {
            dir.mkdir();

            // store fresh copy of dataset in our cache
            File infile = new File(FilenameUtils.concat(dir.getPath(), filename + ".csv"));
            FileUtils.copyURLToFile(new URL(url), infile);

            // grab the data from the original csv file
            RecordReader recordReader = new CSVRecordReader(1);
            recordReader.initialize(new FileSplit(infile));

            // process the original data and build location to fips map
            while (recordReader.hasNext()) {
                List<Writable> row = recordReader.next();
                if (!(inCountyMode ? row.get(3) : row.get(2)).toString().equals("")) {
                    String location = (inCountyMode ? row.get(1) + ", " + row.get(2) : row.get(1)).toString();
                    int fips = (inCountyMode ? row.get(3) : row.get(2)).toInt();
                    locationToFIPS.put(location, fips);
                }
            }
            recordReader.reset();

            // clean the input data
            File cleanDir = cleanData(infile, inCountyMode, recordReader);

            // initiate testing and training process
            testAndTrain(cleanDir, inCountyMode);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private File cleanData(File infile, boolean inCountyMode) {
        File cleanDir = null;
        try {
            // grab the data from the given csv file, assuming no header
            RecordReader recordReader = new CSVRecordReader();
            recordReader.initialize(new FileSplit(infile));
            cleanDir = cleanData(infile, inCountyMode, recordReader);
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }

        return cleanDir;
    }

    private File cleanData(File infile, boolean inCountyMode, RecordReader recordReader) {
        File cleanDir = null;
        try {
            // configure csv input schema
            Schema csvSchema = inCountyMode
                    ? new Schema.Builder()
                    .addColumnsString("date", "county", "state")
                    .addColumnsInteger("fips", "cases", "deaths")
                    .build()
                    : new Schema.Builder()
                    .addColumnsString("date", "state")
                    .addColumnsInteger("fips", "cases", "deaths")
                    .build();

            // clean the data
            TransformProcess tp = new TransformProcess.Builder(csvSchema)
                    .removeAllColumnsExceptFor("date", "fips", "cases")
                    .stringToTimeTransform("date", "YYYY-MM-dd", DateTimeZone.UTC)
                    .convertToDouble("date")
                    .doubleMathOp("date", MathOp.Subtract, 1.5795648E12)
                    .doubleMathOp("date", MathOp.Divide, 8.64E7)
                    .filter(new ConditionFilter(new CategoricalColumnCondition("fips", ConditionOp.Equal, "")))
                    .build();
            TransformProcessRecordReader processedRecordReader = new TransformProcessRecordReader(recordReader, tp);
            processedRecordReader.initialize(new FileSplit(infile));

            // prepare the output files for the clean data
            String cleanDirPath = FilenameUtils.concat(dir.getPath(), "clean_" + infile.getName().replace(".csv", ""));
            cleanDir = new File(cleanDirPath);
            cleanDir.mkdirs();
            RecordWriter writer = new CSVRecordWriter();

            // process the clean data
            while (processedRecordReader.hasNext()) {
                List<Writable> row = processedRecordReader.next();
                int fips = row.get(1).toInt();
                toWrite.put(fips, row);
            }

            // write the clean data
            writeCleanData(writer, toWrite, cleanDirPath);
            toWrite.clear();
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }

        return cleanDir;
    }

    private void writeCleanData(RecordWriter writer, Multimap<Integer, List<Writable>> toWrite, String dirPath) {
        toWrite.asMap().forEach((fips, rows) -> {
            try {
                File out = new File(FilenameUtils.concat(dirPath, fips + ".csv"));
                out.createNewFile();
                writer.initialize(new FileSplit(out), new NumberOfRecordsPartitioner());
                writer.writeBatch(new ArrayList<>(rows));
                writer.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private void testAndTrain(File cleanDir, boolean inCountyMode) {
        // test and train for each location
        List<File> files = Arrays.asList(Objects.requireNonNull(cleanDir.listFiles()));
        files.forEach((file) -> initiateTestTrain(file.getPath()));
    }

    private void initiateTestTrain(String cleanPath) {
        // split data into training and testing sets
        DataSet data = DataSet.createFromFile(cleanPath, numInputs, numOutputs, ",", true);
        data.setColumnNames(new String[]{"date", "fips", "cases"});
        data.setColumnType(1, DataSetColumnType.NOMINAL);
        data.setLabel("cases");

        DataSet[] ttSplit = data.createTrainingAndTestSubsets(0.7, 0.3);
        DataSet trainingData = ttSplit[0];
        DataSet testingData = ttSplit[1];

//        // normalize data
//        Normalizer norm = new ZeroMeanNormalizer(trainingData);
//        norm.normalize(trainingData);
//        norm.normalize(testingData);

        NeuralNetwork model = getModel();

        // train model
        model.learn(trainingData);
        testModel(testingData, model);
    }

    private void testModel(DataSet testingData, NeuralNetwork model) {
        // test model
        MeanSquaredError mse = new MeanSquaredError();
        MeanAbsoluteError mae = new MeanAbsoluteError();
        int lines = 0;

        for (DataSetRow testSetRow : testingData.getRows()) {
            model.setInput(testSetRow.getInput());
            model.calculate();
            double[] prediction = model.getOutput();
            double[] actual = testSetRow.getDesiredOutput();

            mse.addPatternError(prediction, actual);
            mae.addPatternError(prediction, actual);

            if (lines++ < 5) {
                System.out.print("Input: " + Arrays.toString(testSetRow.getInput()));
                System.out.print(" \tOutput: " + Arrays.toString(prediction));
                System.out.println(" \tActual: " + Arrays.toString(actual));
            }
        }

        System.out.println("Mean squared error is: " + mse.getTotalError());
        System.out.println("Mean absolute error is: " + mae.getTotalError());
    }

    private NeuralNetwork getModel() {
        NeuralNetwork model = null;
        try {
            File savedModel = new File(FilenameUtils.concat(dir.getPath(), "model.nnet"));
            if (savedModel.createNewFile()) {
                // configure model
                int numHiddenNodes = 50;
                model = new MultiLayerPerceptron(
                        TransferFunctionType.TANH,
                        numInputs,
                        numHiddenNodes,
                        numHiddenNodes,
                        numHiddenNodes,
                        numOutputs);
                model.setLearningRule(new MomentumBackpropagation());
                MomentumBackpropagation learningRule = (MomentumBackpropagation) model.getLearningRule();
                learningRule.setMaxError(.01);
                learningRule.addListener(this);

                model.save(FilenameUtils.concat(dir.getPath(), "model.nnet"));
            } else {
                model = NeuralNetwork.load(new FileInputStream(savedModel));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return model;
    }

    @Override
    public void handleLearningEvent(LearningEvent learningEvent) {
        MomentumBackpropagation bp = (MomentumBackpropagation) learningEvent.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
    }
}