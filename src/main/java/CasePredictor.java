import org.apache.commons.io.FileUtils;
import org.apache.spark.SparkConf;
import org.datavec.api.records.writer.impl.csv.CSVRecordWriter;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.records.writer.RecordWriter;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.split.partition.NumberOfRecordsPartitioner;
import org.datavec.api.split.partition.Partitioner;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.datavec.spark.transform.misc.StringToWritablesFunction;
import org.datavec.spark.transform.misc.WritablesToStringFunction;
import org.apache.commons.io.FilenameUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.CategoricalColumnCondition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.deeplearning4j.api.storage.StatsStorage;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.api.UIServer;
import org.deeplearning4j.ui.stats.StatsListener;
import org.deeplearning4j.ui.storage.FileStatsStorage;
import org.joda.time.DateTimeZone;
import org.nd4j.evaluation.regression.RegressionEvaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.RmsProp;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CasePredictor {
    private static Map<String, Double> locationToFIPS = new HashMap<>();
    private static int batchSize = 150;
    private static long seed = 123;
    private static int numEpochs = 3;
    private static boolean modelType = true;

    private static MultiLayerNetwork cModel;
    private static File cModelFile;
    private static MultiLayerNetwork sModel;
    private static File sModelFile;

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILE = "covid-19-counties.csv";
    private static final String STATES_FILE = "covid-19-states.csv";

    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
    private static final Logger log = LoggerFactory.getLogger(CasePredictor.class);
    private static final UIServer uiServer = UIServer.getInstance();
    private static final StatsStorage statsStorage = new FileStatsStorage(new File(System.getProperty("java.io.tmpdir"), "ui-stats.dl4j"));

    // configure csv input schema
    private static Schema countySchema = new Schema.Builder()
            .addColumnsString("date", "county", "state")
            .addColumnsInteger("fips", "cases", "deaths")
            .build();
    private static Schema stateSchema = new Schema.Builder()
            .addColumnsString("date", "state")
            .addColumnsInteger("fips", "cases", "deaths")
            .build();

    private static void runModel(String filename, String url) {
        boolean inCountyMode = filename.contains("counties");
        try {
            uiServer.attach(statsStorage);

            File dir = new File(DATA_PATH);
            dir.mkdir();

            // store fresh copy of dataset in our cache
            File infile = new File(DATA_PATH + "src-" + filename);
            FileUtils.copyURLToFile(new URL(url), infile);

            // create output file for the clean data
            File outfile = new File(DATA_PATH + filename);
            if(outfile.exists()){
                outfile.delete();
            }
            outfile.createNewFile();

            // clean the data
            TransformProcess tp = new TransformProcess.Builder(inCountyMode ? countySchema : stateSchema)
                    .removeAllColumnsExceptFor("date", "fips", "cases")
                    .filter(new ConditionFilter(
                            new CategoricalColumnCondition("fips", ConditionOp.Equal, "")
                    ))
                    .stringToTimeTransform("date", "YYYY-MM-DD", DateTimeZone.UTC)
                    .build();

            // grab the data from the original csv file
            RecordReader recordReader = new CSVRecordReader(); // might need to use a sequential record reader
            recordReader.initialize(new FileSplit(infile));

            // prepare the clean data output file
            RecordWriter recordWriter = new CSVRecordWriter();
            Partitioner p = new NumberOfRecordsPartitioner();
            recordWriter.initialize(new FileSplit(outfile), p);

            // process the original data
            List<List<Writable>> originalData = new ArrayList<>();
            while(recordReader.hasNext()){
                List<Writable> row = recordReader.next();
                if ((inCountyMode ? row.get(3) : row.get(2)) != null) {
//                    String location = (inCountyMode ? row.get(1) + ", " + row.get(2) : row.get(1));
//                    double fips = (inCountyMode ? row.get(3) : row.get(2));
//                    locationToFIPS.put(location, fips);
                }
                originalData.add(row);
            }

            // export the clean data
            List<List<Writable>> processedData = LocalTransformExecutor.execute(originalData, tp);
            recordWriter.writeBatch(processedData);
            recordWriter.close();

            // save the data as an ml data set
            int labelIndex = inCountyMode ? 4 : 3;     // the cases column is either the 4th or 5th element
            DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, labelIndex, true);
            DataSet data = iterator.next();

            SplitTestAndTrain ttSplit = data.splitTestAndTrain(0.7); // use 70% of the data for training
            DataSet trainingData = ttSplit.getTrain();
            DataSet testingData = ttSplit.getTest();

            // normalize the data
            DataNormalization normalizer = new NormalizerStandardize(); // might have to fiddle with the normalization
            normalizer.fit(trainingData);
            normalizer.transform(trainingData);
            normalizer.transform(testingData);

            MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new RmsProp.Builder().rmsDecay(0.95).learningRate(1e-2).build())
                    .list()
                    .layer(new LSTM.Builder().name("lstm1")
                            .activation(Activation.TANH).nIn(trainingData.numInputs()).nOut(100).build())
                    .layer(new LSTM.Builder().name("lstm2")
                            .activation(Activation.TANH).nOut(80).build())
                    .layer(new RnnOutputLayer.Builder().name("output")
                            .activation(Activation.SOFTMAX).nOut(trainingData.numOutcomes()).lossFunction(LossFunctions.LossFunction.MSE)
                            .build())
                    .build();

            if (inCountyMode) {
                cModel = new MultiLayerNetwork(conf);
                cModelFile = new File(DATA_PATH, "countyModel.json");
                runModel(cModel, trainingData, testingData, cModelFile);

            } else {
                sModel = new MultiLayerNetwork(conf);
                sModelFile = new File(DATA_PATH, "stateModel.json");
                runModel(sModel, trainingData, testingData, sModelFile);
            }

        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    // take in a county and a date --> predict cases
    // perhaps give a dropdown to select county,
    // and calendar excluding today and past
    private static void makePrediction(String location, String date) {
    }

    private static double getFIPS(String location) {
        return locationToFIPS.get(location);
    }

    private static void runModel(MultiLayerNetwork model, DataSet trainingData, DataSet testingData, File modelFile) {
        try {
            System.out.println(model.summary());

            // train model
            model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(10));
            long startTime = System.currentTimeMillis();
            for (int i = 0; i < numEpochs; i++) {
                model.fit(trainingData);
            }
            long endTime = System.currentTimeMillis();
            System.out.println("=============run time=====================" + (endTime - startTime));

            //evaluate the model on the test set
            RegressionEvaluation eval = new RegressionEvaluation(3);
            INDArray output = model.output(testingData.getFeatures());
            eval.eval(testingData.getLabels(), output);
            log.info(eval.stats());

            // save model to disk
            model.save(modelFile, true);
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