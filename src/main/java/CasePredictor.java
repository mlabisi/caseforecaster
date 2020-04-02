import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.InputSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.joda.time.DateTimeZone;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Nesterovs;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.*;

public class CasePredictor {
    private static Map<String, Integer> locationToFIPS = new HashMap<>();

    private static MultiLayerNetwork cModel;
    private static File cModelFile;
    private static MultiLayerNetwork sModel;
    private static File sModelFile;

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILE = "covid-19-counties.csv";
    private static final String STATES_FILE = "covid-19-states.csv";

    public static final int seed = 12345;
    private static int numEpochs = 15;
    public static final int batchSize = 100;
    public static final double learningRate = 0.01;
    public static final int numInputs = 2;
    private static final int numOutputs = 1;

    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
    private static final Logger log = LoggerFactory.getLogger(CasePredictor.class);
    private static StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));


    private static void runModel(String filename, String url) {
        boolean inCountyMode = filename.contains("counties");
        try {
            File dir = new File(DATA_PATH);
            dir.mkdir();

            // store fresh copy of dataset in our cache
            File infile = new File(DATA_PATH + "src-" + filename);
            FileUtils.copyURLToFile(new URL(url), infile);

            // create output files for the clean data
            File trainOut = new File(DATA_PATH + filename);
            File testOut = new File(DATA_PATH + filename);
            if (trainOut.exists()) {
                trainOut.delete();
            }
            if (testOut.exists()) {
                testOut.delete();
            }
            trainOut.createNewFile();
            testOut.createNewFile();

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
                    .stringToTimeTransform("date", "YYYY-MM-DD", DateTimeZone.UTC)
                    .build();

            // grab the data from the original csv file
            RecordReader recordReader = new CSVRecordReader(); // might need to use a sequential record reader
            recordReader.initialize(new FileSplit(infile));

            // prepare the output files for the clean data
            RecordWriter trainingWriter = new CSVRecordWriter();
            RecordWriter testingWriter = new CSVRecordWriter();
            Partitioner pTrain = new NumberOfRecordsPartitioner();
            Partitioner pTest = new NumberOfRecordsPartitioner();
            trainingWriter.initialize(new FileSplit(trainOut), pTrain);
            testingWriter.initialize(new FileSplit(testOut), pTest);

            // process the original data
            List<List<Writable>> rawTraining = new ArrayList<>();
            List<List<Writable>> rawTesting = new ArrayList<>();
            recordReader.next();

            while (recordReader.hasNext()) {
                List<Writable> row = recordReader.next();
                if (!(inCountyMode ? row.get(3) : row.get(2)).toString().equals("")) {
                    String location = (inCountyMode ? row.get(1) + ", " + row.get(2) : row.get(1)).toString();
                    int fips = (inCountyMode ? row.get(3) : row.get(2)).toInt();
                    locationToFIPS.put(location, fips);
                }
                rawTraining.add(row);
            }

            for (int i = 0; i < (.07 * rawTraining.size()); i++) {
                rawTesting.add(rawTraining.remove(i));
            }

            // export the clean data
            List<List<Writable>> processedTrain = LocalTransformExecutor.execute(rawTraining, tp);
            List<List<Writable>> processedTest = LocalTransformExecutor.execute(rawTesting, tp);
            trainingWriter.writeBatch(processedTrain);
            testingWriter.writeBatch(processedTest);
            trainingWriter.close();
            testingWriter.close();
            System.out.println("Successfully wrote clean data to " + trainOut.getPath() + " and \n" + testOut.getPath());

            // save clean data as ml data sets
            RecordReader trainingReader = new CSVRecordReader();
            RecordReader testingReader = new CSVRecordReader();
            trainingReader.initialize(new FileSplit(trainOut));
            testingReader.initialize(new FileSplit(testOut));

            int labelIndex = 2;
            DataSetIterator trainingData = new RecordReaderDataSetIterator(trainingReader, batchSize, labelIndex, labelIndex, true);
            DataSetIterator testingData = new RecordReaderDataSetIterator(testingReader, batchSize, labelIndex, labelIndex, true);

            // configure the neural network
            final int numHiddenNodes = 50;
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new Nesterovs(learningRate, 0.9))
                    .list()
                    .layer(new DenseLayer.Builder().nIn(numInputs).nOut(numHiddenNodes)
                            .activation(Activation.TANH).build())
//                    .layer(new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes)
//                            .activation(Activation.TANH).build())
                    .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .nIn(numHiddenNodes).nOut(numOutputs).build())
                    .build();

            // run the appropriate model
            if (inCountyMode) {
                cModel = new MultiLayerNetwork(conf);
                cModelFile = new File(DATA_PATH, "countyModel.json");
                runModel(cModel, cModelFile, testingData, trainingData);

            } else {
                sModel = new MultiLayerNetwork(conf);
                sModelFile = new File(DATA_PATH, "stateModel.json");
                runModel(sModel, sModelFile, testingData, trainingData);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void runModel(MultiLayerNetwork model, File modelFile, DataSetIterator testingData, DataSetIterator trainingData) {
        try {
            // initialize model
            model.init();

            // connect to ui
            model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

            // train the model
            for (int i = 0; i < numEpochs; i++) {
                model.fit(trainingData);
            }

            // save model to disk
            model.save(modelFile, true);

            // test the model
            System.out.println("cases: \t\tpredicted:");
            RegressionEvaluation eval = new RegressionEvaluation(1);
            while (testingData.hasNext()) {
                DataSet row = testingData.next();
                INDArray features = row.getFeatures();
                INDArray cases = row.getLabels();

                List<INDArray> predicted = model.feedForward(features);

                for (INDArray prediction :
                        predicted) {
                    System.out.println(cases.getInt(0) + "\t\t\t" + prediction.getInt(0));
                    eval.eval(cases, prediction);
                }
            }
            System.out.println(eval.stats());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void run() {
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
        runModel(STATES_FILE, STATES_URL);

//        UIServer uiServer = UIServer.getInstance();
//        uiServer.attach(statsStorage);
//        System.out.println("UI Server running at " + uiServer.getAddress());
    }

    // take in a county and a date --> predict cases
    // perhaps give a dropdown to select county,
    // and calendar excluding today and past
    private static void makePrediction(String location, String date) {
    }

    private static double getFIPS(String location) {
        return locationToFIPS.get(location);
    }

    public static void main(String[] args) {
        run();

//        makePrediction("Snohomish, Washington", "2020-04-03");
//        makePrediction("Los Angeles, California", "2020-04-03");
//        makePrediction("Hawaii", "2020-04-03");
//        makePrediction("New York", "2020-04-03");

        while (true) {
            System.out.print("");
        }

        // TODO: GUI for makePrediction
    }
}