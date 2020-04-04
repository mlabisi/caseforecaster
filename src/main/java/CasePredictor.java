import com.google.common.collect.ArrayListMultimap;
import com.google.common.collect.HashMultimap;
import com.google.common.collect.Lists;
import com.google.common.collect.Multimap;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.SequenceRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.reader.impl.csv.CSVSequenceRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.records.reader.impl.transform.TransformProcessRecordReader;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.api.split.FileSplit;
import org.datavec.api.split.NumberedFileInputSplit;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.Writable;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.datavec.SequenceRecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerMinMaxScaler;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.*;

import static org.joda.time.DateTimeZone.*;

public class CasePredictor {
    private static Map<String, Integer> locationToFIPS = new HashMap<>();
    private static Multimap<Integer, List<Writable>> trainToWrite = ArrayListMultimap.create();
    private static Multimap<Integer, List<Writable>> testToWrite = ArrayListMultimap.create();

    private static MultiLayerNetwork cModel;
    private static File cModelFile;
    private static MultiLayerNetwork sModel;
    private static File sModelFile;

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILE = "covid-19-counties.csv";
    private static final String STATES_FILE = "covid-19-states.csv";

    public static final int seed = 12345;
    public static final int batchSize = 100;
    public static final double learningRate = 0.01;
    public static final int numInputs = 2;
    private static final int numOutputs = 1;
    private static final int labelIndex = 2;


    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
    private static StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));
    private static File dir = new File(DATA_PATH);
    private static File countiesDirTrain = new File(DATA_PATH + "counties-train/");
    private static File countiesDirTest = new File(DATA_PATH + "counties-test/");
    private static File statesDirTrain = new File(DATA_PATH + "states-train/");
    private static File statesDirTest = new File(DATA_PATH + "states-test/");


    private static void runModel(String filename, String url) {
        boolean inCountyMode = filename.contains("counties");
        try {
            dir.mkdir();
            countiesDirTrain.mkdir();
            countiesDirTest.mkdir();
            statesDirTrain.mkdir();
            statesDirTest.mkdir();

            // store fresh copy of dataset in our cache
            File infile = new File(DATA_PATH + "src-" + filename);
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

            // clean the data
            TransformProcess tp = new TransformProcess.Builder(csvSchema)
                    .removeAllColumnsExceptFor("date", "fips", "cases")
                    .filter(new ConditionFilter(new CategoricalColumnCondition("fips", ConditionOp.InSet, new HashSet<>(Collections.singletonList("")))))
                    .stringToTimeTransform("date", "YYYY-MM-DD", UTC)
                    .build();


            // grab the data from the original csv file
            RecordReader recordReader = new CSVRecordReader(1); // might need to use a sequential record reader
            recordReader.initialize(new FileSplit(infile));
            TransformProcessRecordReader processRecordReader = new TransformProcessRecordReader(recordReader, tp);
            processRecordReader.initialize(new FileSplit(infile));

            // process the original data and build location to fips map
            // grab total number of rows to calculate training-testing split
            int setSize = 0;
            while (recordReader.hasNext()) {
                List<Writable> row = recordReader.next();
                if (!(inCountyMode ? row.get(3) : row.get(2)).toString().equals("")) {
                    String location = (inCountyMode ? row.get(1) + ", " + row.get(2) : row.get(1)).toString();
                    int fips = (inCountyMode ? row.get(3) : row.get(2)).toInt();
                    locationToFIPS.put(location, fips);
                }
                setSize++;
            }
            recordReader.reset();

            // prepare the output files for the clean data
            RecordWriter trainingWriter = new CSVRecordWriter();
            RecordWriter testingWriter = new CSVRecordWriter();
            int ct = 0;

            // process the clean data
            while (processRecordReader.hasNext()) {
                List<Writable> row = processRecordReader.next();
                int fips = row.get(1).toInt();

                // initialize file for clean data
                if (ct++ < (0.7 * setSize)) {
                    trainToWrite.put(fips, row);
                } else {
                    testToWrite.put(fips, row);
                }
            }

            collectRows(inCountyMode, trainingWriter, trainToWrite, countiesDirTrain, statesDirTrain);
            collectRows(inCountyMode, testingWriter, testToWrite, countiesDirTest, statesDirTest);

            // save clean data as ml data sets
            SequenceRecordReader trainingReader = new CSVSequenceRecordReader(1);
            trainingReader.initialize(new FileSplit(new File((inCountyMode ? countiesDirTrain.getPath() : statesDirTrain.getPath()) + "/")));
            SequenceRecordReader testingReader = new CSVSequenceRecordReader(1);
            testingReader.initialize(new FileSplit(new File((inCountyMode ? countiesDirTest.getPath() : statesDirTest.getPath()) + "/")));

            DataSetIterator trainingIter = new SequenceRecordReaderDataSetIterator(trainingReader, batchSize, -1, labelIndex, true);
            DataSetIterator testingIter = new SequenceRecordReaderDataSetIterator(testingReader, batchSize, -1, labelIndex, true);

            // Create data set
            DataSet trainingData = trainingIter.next();
            DataSet testingData = testingIter.next();

            //Normalize data, including labels (fitLabel=true)
            NormalizerMinMaxScaler normalizer = new NormalizerMinMaxScaler(0, 1);
            normalizer.fitLabel(true);
            normalizer.fit(trainingData); //Collect training data statistics

            normalizer.transform(trainingData);
            normalizer.transform(testingData);

            // configure the neural network
            final int numHiddenNodes = 50;
            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .list()
                    .layer(0, new LSTM.Builder().activation(Activation.TANH).nIn(numInputs).nOut(numHiddenNodes).build())
                    .layer(1, new DenseLayer.Builder().nIn(numHiddenNodes).nOut(numHiddenNodes).weightInit(WeightInit.XAVIER).activation(Activation.RELU).build()) //Configuring Layers
                    .layer(2, new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY).nIn(numHiddenNodes).nOut(numOutputs).build())
                    .build();

            // run the appropriate model
            if (inCountyMode) {
                cModel = new MultiLayerNetwork(conf);
                cModelFile = new File(DATA_PATH, "countyModel.json");
                runModel(cModel, cModelFile, testingData, trainingData, normalizer);

            } else {
                sModel = new MultiLayerNetwork(conf);
                sModelFile = new File(DATA_PATH, "stateModel.json");
                runModel(sModel, sModelFile, testingData, trainingData, normalizer);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private static void collectRows(boolean inCountyMode, RecordWriter testingWriter, Multimap<Integer, List<Writable>> testToWrite, File countiesDirTest, File statesDirTest) {
        testToWrite.asMap().forEach((fips, rows) -> {
            try {
                File out = new File((inCountyMode ? countiesDirTest.getPath() : statesDirTest.getPath()) + "/" + fips + ".csv");
                out.createNewFile();
                testingWriter.initialize(new FileSplit(out), new NumberOfRecordsPartitioner());
                testingWriter.writeBatch(new ArrayList<>(rows));
                testingWriter.close();
            } catch (Exception e) {
                e.printStackTrace();
            }
        });
    }

    private static void runModel(MultiLayerNetwork model, File modelFile, DataSet testingData, DataSet trainingData, NormalizerMinMaxScaler normalizer) {
        // initialize model
        model.init();

        // connect to ui
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));

        // train the model
        int numEpochs = 15;
        for (int i = 0; i < numEpochs; i++) {
            model.fit(trainingData);
            System.out.println("Epoch " + i + " complete. Time series evaluation:");

            RegressionEvaluation evaluation = new RegressionEvaluation(1);
            INDArray features = testingData.getFeatures();
            INDArray labels = testingData.getLabels();

            INDArray predicted = model.output(features, false);

//            System.out.println("PREDICTED\n" + predicted);
//            System.out.println("ACTUAL\n" + labels);

            evaluation.evalTimeSeries(labels, predicted);
            System.out.println(evaluation.stats());
        }

        // init rrnTimeStep with train data and predict test data
        model.rnnTimeStep(trainingData.getFeatures());
        INDArray predicted = model.rnnTimeStep(testingData.getFeatures());

        // revert data back to original values for plotting
        normalizer.revert(trainingData);
        normalizer.revert(testingData);
        normalizer.revertLabels(predicted);

        //Create plot with out data
        XYSeriesCollection c = new XYSeriesCollection();
        createSeries(c, trainingData.getFeatures(), 0, "Train data");
        createSeries(c, testingData.getFeatures(), 99, "Actual test data");
        createSeries(c, predicted, 100, "Predicted test data");
    }

    private static XYSeriesCollection createSeries(XYSeriesCollection seriesCollection, INDArray data, int offset, String name) {
        int nRows = (int) data.shape()[2];
        XYSeries series = new XYSeries(name);
        for (int i = 0; i < nRows; i++) {
            series.add(i + offset, data.getDouble(i));
        }

        seriesCollection.addSeries(series);

        return seriesCollection;
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