import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.feature.*;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.regression.DecisionTreeRegressor;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.Map;

public class CaseForecaster {
    private static LinearRegressionModel countyModel;
    private static LinearRegressionModel stateModel;

    private static SparkSession spark = SparkSession.builder().master("local[*]").appName("Case Predictor").getOrCreate();
    private static Map<String, Integer> fipsToLocation = new HashMap<>();

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
    private static final String COUNTIES_FILE = "covid-19-counties.csv";
    private static final String STATES_FILE = "covid-19-states.csv";

//    private static void trainModels() {
//        // gather and process county data
//        Dataset<Row> cDataFrame = gatherData(COUNTIES_FILE, COUNTRIES_URL);
//        int countyRows = processData(COUNTIES_FILE, cDataFrame);
//
//        // gather and process state data
//        Dataset<Row> sDataFrame = gatherData(STATES_FILE, STATES_URL);
//        int stateRows = processData(STATES_FILE, sDataFrame);
//
//        // prepare training and testing data
////        Dataset<Row> cTraining = spark.createDataFrame(cTrainingData, schema);
////        Dataset<Row> cTesting = spark.createDataFrame(cTestingData, schema);
////
////        Dataset<Row> sTraining = spark.createDataFrame(sTrainingData, schema);
////        Dataset<Row> sTesting = spark.createDataFrame(sTestingData, schema);
////
////        // define the linear regression
////        LinearRegression linReg = new LinearRegression().setFeaturesCol("features").setLabelCol("cases");
////
////        // train models
////        countyModel = linReg.fit(cTraining);
////        stateModel = linReg.fit(sTraining);
////
////        // test models
////        testModel(cTesting, countyModel);
////        testModel(sTesting, stateModel);
//    }

//    private static Dataset<Row> gatherData(String filename, String url) {
//        System.out.println("Generating data from given source: " + url);
//        boolean inCountyMode = filename.contains("counties");
//        Dataset<Row> data = null;
//        try {
//            File dir = new File(DATA_PATH);
//            dir.mkdir();
//
//            // store fresh copy of dataset in our cache
//            File file = new File(DATA_PATH + filename);
//            FileUtils.copyURLToFile(new URL(url), file);
//            System.out.println("Successfully created " + DATA_PATH + filename);
//
//            // load the data as a Dataframe
//            data = spark.read().format("csv").option("header", true).load(DATA_PATH + filename);
////            data = data.drop("deaths");
//
//            FeatureHasher hasher = new FeatureHasher()
//                    .setInputCols(inCountyMode ? new String[]{"date", "county", "state", "fips"} : new String[]{"date", "state", "fips"})
//                    .setOutputCol("features");
//
//            data = hasher.transform(data);
//            data.show(false);
//        } catch (MalformedURLException e) {
//            System.out.println("Erroneous URL provided.");
//            e.printStackTrace();
//        } catch (IOException e) {
//            System.out.println("File could not be created.");
//            e.printStackTrace();
//        }
//
//        return data;
//    }
//
//    private static int processData(String filename, Dataset<Row> dataframe) {
//        // define decision tree regression
//        DecisionTreeRegressor decTree = new DecisionTreeRegressor().setFeaturesCol("features")/*.setLabelCol("cases")*/;
//
//        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[]{decTree});
//        int rows = 0;
////        try (final BufferedReader reader = Files.newBufferedReader(Paths.get(DATA_PATH + filename), StandardCharsets.UTF_8)) {
////            boolean inCountyMode = filename.contains("counties");
////            Dataset<Row>[] splits = dataframe.randomSplit(new double[]{0.7, 0.3});
////            Dataset<Row> trainingData = splits[0];
////            Dataset<Row> testingData = splits[1];
////
////            StructType schema = new StructType(new StructField[]{
////                    new StructField("cases", DataTypes.DoubleType, false, Metadata.empty()),
////                    new StructField("features", new VectorUDT(), false, Metadata.empty())
////            });
////
////            Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(reader);
////            for (CSVRecord record : records) {
////                if (record.get("fips").equals("")) {
////                    continue;
////                }
////
////                double date = Integer.parseInt(record.get("date").replaceAll("-", ""));
////                double fips = Integer.parseInt(record.get("fips"));
////                double cases = Integer.parseInt(record.get("cases"));
////
////                String location = (inCountyMode ? record.get("county") + ", " : "") + record.get("state");
////                fipsToLocation.put(location, fips);
////
//////                if(++rows > (inCountyMode ? 18000 : 1300) ) {
//////                    testingData.add(RowFactory.create(cases, Vectors.dense(fips, date)));
//////                } else {
//////                    trainingData.add(RowFactory.create(cases, Vectors.dense(fips, date)));
//////                }
////            }
////        } catch (IOException e) {
////            e.printStackTrace();
////        }
//        return rows;
//    }
//
//    private static void testModel(Dataset<Row> testing, LinearRegressionModel model) {
//        Dataset<Row> sResults = model.transform(testing);
////        Dataset<Row> sEntries = sResults.select("features", "cases", "prediction");
////        for (Row entry: sEntries.collectAsList()) {
////            System.out.println("(" + entry.get(0) + ", " + entry.get(1) + ") -> prediction=" + entry.get(2));
////        }
//    }
//
//    private static void makePrediction(String location, String givenDate) {
//        // take in a county and a date --> predict cases
//        // perhaps give a dropdown to select county,
//        // and calendar excluding today and past
//        double fips = getFIPS(location);
//        double date = Integer.parseInt(givenDate.replaceAll("-", ""));
//        Vector input = Vectors.dense(fips, date);
//
//        int casesCt = (int) (location.contains(",") ? countyModel.predict(input) : stateModel.predict(input));
//        System.out.println("I predict there will be " + casesCt + " cases in " + location + " on " + givenDate + ".");
//    }
//
//    private static double getFIPS(String location) {
//        return fipsToLocation.get(location);
//    }

    private static void gatherData(String filename, String url) {
        System.out.println("Generating data from given source: " + url);
        boolean inCountyMode = filename.contains("counties");
        Dataset<Row> data = null;
        try {
            File dir = new File(DATA_PATH);
            dir.mkdir();

            // store fresh copy of dataset in our cache
            File file = new File(DATA_PATH + filename);
            FileUtils.copyURLToFile(new URL(url), file);
            System.out.println("Successfully created " + DATA_PATH + filename);

            // configure csv input schema
            StructType csvSchema = new StructType(
                    inCountyMode ?
                            new StructField[]{
                                    new StructField("date", DataTypes.StringType, false, Metadata.empty()),
                                    new StructField("county", DataTypes.StringType, false, Metadata.empty()),
                                    new StructField("state", DataTypes.StringType, false, Metadata.empty()),
                                    new StructField("fips", DataTypes.IntegerType, false, Metadata.empty()),
                                    new StructField("cases", DataTypes.IntegerType, false, Metadata.empty()),
                                    new StructField("deaths", DataTypes.IntegerType, false, Metadata.empty())
                            }
                            : new StructField[]{
                            new StructField("date", DataTypes.StringType, false, Metadata.empty()),
                            new StructField("state", DataTypes.StringType, false, Metadata.empty()),
                            new StructField("fips", DataTypes.IntegerType, false, Metadata.empty()),
                            new StructField("cases", DataTypes.IntegerType, false, Metadata.empty()),
                            new StructField("deaths", DataTypes.IntegerType, false, Metadata.empty())
                    });

            // load the data as a Dataframe
            data = spark.read().format("csv").option("header", true).schema(csvSchema).load(DATA_PATH + filename);

            // build up location to FIPS index
            data.foreach(row -> {
                if ((inCountyMode ? row.get(3) : row.get(2)) != null) {
                    String location = (inCountyMode ? row.getString(1) + ", " + row.getString(2) : row.getString(1));
                    int fips = (inCountyMode ? row.getInt(3) : row.getInt(2));
                    fipsToLocation.put(location, fips);
                }
            });

            // remove location since we have the fips
            data.drop("state");
            if (inCountyMode) {
                data.drop("county");
            }

            // split all data into training data and testing data
            Dataset<Row>[] splits = data.randomSplit(new double[]{0.7, 0.3});
            Dataset<Row> trainingData = splits[0];
            Dataset<Row> testingData = splits[1];

            // configure pipeline
            StringIndexer indexer = new StringIndexer()
                    .setInputCol("date")
                    .setOutputCol("indexedDate")
                    .setHandleInvalid("skip");
            VectorAssembler assembler = new VectorAssembler()
                    .setInputCols(new String[]{"indexedDate", "fips"})
                    .setOutputCol("features")
                    .setHandleInvalid("skip");
            LinearRegression dtReg = new LinearRegression()
                    .setFeaturesCol("features")
                    .setLabelCol("cases");
            Pipeline pipeline = new Pipeline()
                    .setStages(new PipelineStage[]{indexer, assembler, dtReg});

            // train model
            PipelineModel model = pipeline.fit(trainingData);

            // test model
            Dataset<Row> predictions = model.transform(testingData);
            for (Row entry : predictions.select("features", "cases").collectAsList()) {
                System.out.println("(" + entry.get(0) + ") -> predicted cases=" + entry.get(1));
            }
        } catch (MalformedURLException e) {
            System.out.println("Erroneous URL provided.");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("File could not be created.");
            e.printStackTrace();
        }
    }

    private static void runModels() {
        // gather data
        // prepare data
        // collect training data
        // configure pipeline tokenizer
        // configure pipeline transformer
        // configure pipeline model
        // fit training data
        // collect testing data
        // predict using testing data
        gatherData(COUNTIES_FILE, COUNTRIES_URL);
        gatherData(STATES_FILE, STATES_URL);
    }

    public static void main(String[] args) {
        runModels();
//
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