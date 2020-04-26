import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.transform.transform.time.StringToTimeTransform;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.util.ModelSerializer;
import org.joda.time.DateTimeZone;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CasePredictor {
    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILE = "covid_19_counties";
    private static final String STATES_FILE = "covid_19_states";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("user.dir") + "/src/main", "resources");

    private static Map<String, Integer> locationToFIPS = new HashMap<>();
    private static List<List<Writable>> toWrite = new ArrayList<>();
    private static File rscDir = new File(DATA_PATH);

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
//        buildModel(COUNTIES_FILE, COUNTRIES_URL);
        // train and test the model using state data
        buildModel(STATES_FILE, STATES_URL);
    }

//    private void makePrediction(String location, String date) {
//        // perhaps give a dropdown to select county,
//        // and calendar excluding today and past
//        // take in a county and a date --> predict cases);
//        boolean inCountyMode = location.contains(",");
//        File rawInput = new File(FilenameUtils.concat(dir.getPath(), "input.csv"));
//        try (FileWriter writer = new FileWriter(rawInput)) {
//            int fips = locationToFIPS.get(location);
//            writer.write(fips + ",0");
//            File cleanInput = cleanData(rawInput, inCountyMode);
//            DataSet input = DataSet.createFromFile(cleanInput.getPath(), numInputs, numOutputs, ",");
//            input.setColumnNames(new String[]{"date", "fips", "cases"});
//            input.setColumnType(0, DataSetColumnType.NOMINAL);
//            input.setLabel("cases");
//
//            MultiLayerNetwork model = getModel("" + fips);
//            model.setInput(input.getRowAt(0).getInput());
//            model.calculate();
//            double[] prediction = model.getOutput();
//            System.out.println("I predict there will be " + prediction[0] + " cases in " + location + " on " + date + ".");
//        } catch (Exception e) {
//            e.printStackTrace();
//        }
//
//    }

    private void buildModel(String filename, String url) {
        boolean inCountyMode = filename.contains("counties");
        try {
            rscDir.mkdir();

            // store fresh copy of dataset in our cache
            File infile = new File(FilenameUtils.concat(rscDir.getPath(), filename + ".csv"));
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
            File cleanFile = cleanData(infile, inCountyMode, recordReader);

            // initiate testing and training process
            initiateTestTrain(cleanFile);
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
        File cleanFile = null;
        try {
            String fips = inCountyMode ? "53061" : "53";

            // configure csv input schema
            Schema csvSchema = inCountyMode
                    ? new Schema.Builder()
                    .addColumnsString("date", "county", "state", "fips")
                    .addColumnsInteger("cases", "deaths")
                    .build()
                    : new Schema.Builder()
                    .addColumnsString("date", "state", "fips")
                    .addColumnsInteger("cases", "deaths")
                    .build();

            // create the transformation to be applied on raw data
            TransformProcess tp = new TransformProcess.Builder(csvSchema)
                    .removeAllColumnsExceptFor("date", "fips", "cases")
                    .filter(new ConditionFilter(new CategoricalColumnCondition("fips", ConditionOp.Equal, "")))
                    .transform(new StringToTimeTransform("date", "YYYY-MM-dd", DateTimeZone.UTC))
                    .convertToSequence("fips", new NumericalColumnComparator("date", true))
                    .build();

            // now read in the raw data to memory using spark
            SparkConf conf = new SparkConf().setMaster("local[*]").setAppName("Case Predictor");
            JavaSparkContext sc = new JavaSparkContext(conf);
            JavaRDD<String> rawData = sc.textFile(infile.getPath()).filter(row -> !row.startsWith("date"));

            RecordReader reader = new CSVRecordReader();
            StringToWritablesFunction swFunc = new StringToWritablesFunction(reader);
            JavaRDD<List<Writable>> parsed = (JavaRDD<List<Writable>>) rawData.map(swFunc);

            // execute the transform
            JavaRDD<List<List<Writable>>> processedData = SparkTransformExecutor.executeToSequence(parsed, tp);

            // save the clean data
            String cleanPath = FilenameUtils.concat(rscDir.getPath(), "clean_" + infile.getName().replace(".csv", ""));
            processedData.saveAsTextFile(cleanPath);

            // TODO: make each sequence the same length (https://deeplearning4j.konduit.ai/getting-started/tutorials/advanced-autoencoder#examine-sequence-lengths)
            //       read in the sequences as TIME SERIES using EQUAL LENGTH (https://deeplearning4j.org/api/latest/org/datavec/api/records/reader/impl/csv/CSVMultiSequenceRecordReader.html)
            //       create training and testing sets
            //       train, test, plot

            TransformProcessRecordReader processedRecordReader = new TransformProcessRecordReader(recordReader, tp);
            processedRecordReader.initialize(new FileSplit(infile));


            RecordWriter writer = new CSVRecordWriter();

            // process the clean data
            while (processedRecordReader.hasNext()) {
                List<Writable> row = processedRecordReader.next();
                toWrite.add(row);
            }

            // write the clean data
            writeCleanData(writer, toWrite, cleanPath);
            toWrite.clear();
        } catch (InterruptedException | IOException e) {
            e.printStackTrace();
        }

        return cleanFile;
    }

    private void writeCleanData(RecordWriter writer, List<List<Writable>> toWrite, String outPath) {
        try {
            File out = new File(outPath);
            out.createNewFile();
            writer.initialize(new FileSplit(out), new NumberOfRecordsPartitioner());
            writer.writeBatch(toWrite);
            writer.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void initiateTestTrain(File file) {
        try {
            // read in data
            SequenceRecordReader reader = new CSVSequenceRecordReader();
            reader.initialize(new FileSplit(file));
            DataSetIterator iter = new SequenceRecordReaderDataSetIterator(reader, 50, 1, 2, true);
            DataSet data = iter.next();

            // normalize training data, then set preprocessor
            SplitTestAndTrain ttSplit = data.splitTestAndTrain(0.7);
            DataSet trainingData = ttSplit.getTrain();

//                NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler();
//                normalizer.fit(trainingData);
//                iter.setPreProcessor(normalizer);
//
//                // split data into training and testing sets
//                ttSplit = data.splitTestAndTrain(0.7);
//                trainingData = ttSplit.getTrain();
            DataSet testingData = ttSplit.getTest();

            // build or grab the model
            MultiLayerNetwork model = getModel();

            // train model
            for (int i = 0; i < 1000; i++) {
                model.fit(trainingData);
                model.rnnClearPreviousState();
            }

            // save model
            ModelSerializer.writeModel(model, FilenameUtils.concat(rscDir.getPath(), "model.zip"), true);

            // test model
            testModel(testingData, model);
        } catch (IOException | InterruptedException e) {
            e.printStackTrace();
        }
    }

    private void testModel(DataSet testingData, MultiLayerNetwork model) {
        double[] predicted = new double[testingData.numExamples()];
        double[] actual = new double[testingData.numExamples()];

        for (int i = 0; i < testingData.numExamples(); i++) {
            predicted[i] = model.rnnTimeStep(testingData.get(i).getFeatures()).getDouble(0);
            actual[i] = testingData.get(i).getLabels().getDouble(0);
        }

        for (int i = 0; i < predicted.length; i++) {
            System.out.println(predicted[i] + "," + actual[i]);
        }
        PlotUtil.plot(predicted, actual);
        System.out.println("done");
    }

    private MultiLayerNetwork getModel() {
        MultiLayerNetwork model = null;
        try {
            File savedModel = new File(FilenameUtils.concat(rscDir.getPath(), "model.zip"));
            if (!savedModel.exists()) {
                // configure model
                int seed = 12345;
                int nIn = 2;
                int nOut = 1;
                int lstmLayer1Size = 256;
                int lstmLayer2Size = 256;
                int denseLayerSize = 32;
                double dropoutRatio = 0.2;
                int truncatedBPTTLength = 22;

                MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                        .seed(seed)
                        .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                        .weightInit(WeightInit.XAVIER)
                        .weightDecay(1e-4)
                        .list()
                        .layer(0, new LSTM.Builder()
                                .nIn(nIn)
                                .nOut(lstmLayer1Size)
                                .activation(Activation.TANH)
                                .gateActivationFunction(Activation.HARDSIGMOID)
                                .dropOut(dropoutRatio)
                                .build())
                        .layer(1, new LSTM.Builder()
                                .nIn(lstmLayer1Size)
                                .nOut(lstmLayer2Size)
                                .activation(Activation.TANH)
                                .gateActivationFunction(Activation.HARDSIGMOID)
                                .dropOut(dropoutRatio)
                                .build())
                        .layer(2, new DenseLayer.Builder()
                                .nIn(lstmLayer2Size)
                                .nOut(denseLayerSize)
                                .activation(Activation.RELU)
                                .build())
                        .layer(3, new RnnOutputLayer.Builder()
                                .nIn(denseLayerSize)
                                .nOut(nOut)
                                .activation(Activation.IDENTITY)
                                .lossFunction(LossFunctions.LossFunction.MSE)
                                .build())
                        .backpropType(BackpropType.TruncatedBPTT)
                        .tBPTTForwardLength(truncatedBPTTLength)
                        .tBPTTBackwardLength(truncatedBPTTLength)
                        .build();

                model = new MultiLayerNetwork(conf);
                model.init();
                model.setListeners(new ScoreIterationListener(100));
            } else {
                model = ModelSerializer.restoreMultiLayerNetwork(FilenameUtils.concat(rscDir.getPath(), "model.zip"));
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return model;
    }
}