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
import org.neuroph.util.data.norm.Normalizer;
import org.neuroph.util.data.norm.RangeNormalizer;

import java.io.File;
import java.net.URL;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

import static org.joda.time.DateTimeZone.UTC;

public class CasePredictor implements LearningEventListener {

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILE = "covid_19_counties.csv";
    private static final String STATES_FILE = "covid_19_states.csv";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime");

    private static final double learningRate = 0.01;
    private static final int numInputs = 2;
    private static final int numOutputs = 1;

    private static Map<String, Integer> locationToFIPS = new HashMap<>();
    private static File dir = new File(DATA_PATH);

    private void runModel(String filename, String url) {
        boolean inCountyMode = filename.contains("counties");
        try {
            dir.mkdir();

            // store fresh copy of dataset in our cache
            File infile = new File(DATA_PATH + "/" + filename);
            FileUtils.copyURLToFile(new URL(url), infile);

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

            // clean the data
            TransformProcess tp = new TransformProcess.Builder(csvSchema)
                    .removeAllColumnsExceptFor("date", "fips", "cases")
                    .stringToTimeTransform("date", "YYYY-MM-DD", UTC)
                    .timeMathOp("date", MathOp.Add, (long)1.5795648e12 , TimeUnit.SECONDS)
                    .filter(new ConditionFilter(new CategoricalColumnCondition("fips", ConditionOp.Equal, "")))
                    .build();
            TransformProcessRecordReader processedRecordReader = new TransformProcessRecordReader(recordReader, tp);
            processedRecordReader.initialize(new FileSplit(infile));

            String cleanFilename = dir + "/clean_" + filename;

            // write out the clean data
            RecordWriter cleanWriter = new CSVRecordWriter();
            File out = new File(cleanFilename);
            out.createNewFile();
            cleanWriter.initialize(new FileSplit(out), new NumberOfRecordsPartitioner());
            while (processedRecordReader.hasNext()) {
                cleanWriter.write(processedRecordReader.next());
            }
            cleanWriter.close();

            // split data into training and testing sets
            DataSet data = DataSet.createFromFile(cleanFilename, numInputs, numOutputs, ",");
            data.setColumnNames(new String[]{"date", "fips", "cases"});
            data.setColumnType(1, DataSetColumnType.NOMINAL);
            data.setLabel("cases");

            DataSet[] ttSplit = data.createTrainingAndTestSubsets(0.7, 0.3);
            DataSet trainingData = ttSplit[0];
            DataSet testingData = ttSplit[1];

            // normalize data
            Normalizer norm = new RangeNormalizer(0, 500);
            norm.normalize(trainingData);
            norm.normalize(testingData);

            // configure model
            int numHiddenNodes = 50;

            MultiLayerPerceptron model = new MultiLayerPerceptron(
                    TransferFunctionType.TANH,
                    numInputs,
                    numHiddenNodes,
                    numHiddenNodes,
                    numHiddenNodes,
                    numOutputs);
            model.setLearningRule(new MomentumBackpropagation());
            MomentumBackpropagation learningRule = (MomentumBackpropagation) model.getLearningRule();
            learningRule.setMaxError(10);
            learningRule.addListener(this);

            // train model
            model.learn(trainingData);

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
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void run() {
        // gather data
        // prepare data
        // collect training data
        // configure pipeline tokenizer
        // configure pipeline transformer
        // configure pipeline model
        // fit training data
        // collect testing data
        // predict using testing data
        runModel(COUNTIES_FILE, COUNTRIES_URL);
//        runModel(STATES_FILE, STATES_URL, statesDirTrain, statesDirTest);
    }

    public static void main(String[] args) {
        (new CasePredictor()).run();

//        makePrediction("Snohomish, Washington", "2020-04-03");
//        makePrediction("Los Angeles, California", "2020-04-03");
//        makePrediction("Hawaii", "2020-04-03");
//        makePrediction("New York", "2020-04-03");

        while (true) {
            System.out.print("");
        }

        // TODO: GUI for makePrediction
    }

    @Override
    public void handleLearningEvent(LearningEvent learningEvent) {
        MomentumBackpropagation bp = (MomentumBackpropagation) learningEvent.getSource();
        System.out.println(bp.getCurrentIteration() + ". iteration | Total network error: " + bp.getTotalNetworkError());
    }
}