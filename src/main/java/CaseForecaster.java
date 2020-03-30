import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.ml.linalg.VectorUDT;
import org.apache.spark.ml.linalg.Vectors;
import org.apache.spark.ml.regression.LinearRegression;
import org.apache.spark.ml.regression.LinearRegressionModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.RowFactory;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.types.Metadata;
import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class CaseForecaster {
    private static LinearRegressionModel countyModel;
    private static LinearRegressionModel stateModel;
    private static SparkSession spark = SparkSession.builder().master("local[*]").appName("Case Predictor").getOrCreate();
    private static Map<String, Double> fipsToLocation = new HashMap<>();

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
    private static final String COUNTIES_FILE = "covid-19-counties.csv";
    private static final String STATES_FILE = "covid-19-states.csv";

    private static void trainModels() {
        List<Row> cTrainingData = new ArrayList<>();
        List<Row> cTestingData = new ArrayList<>();

        List<Row> sTrainingData = new ArrayList<>();
        List<Row> sTestingData = new ArrayList<>();

        // gather and process county data
        gatherData(COUNTIES_FILE, COUNTRIES_URL);
        int countyRows = processData(COUNTIES_FILE, cTrainingData, cTestingData);

        // gather and process state data
        gatherData(STATES_FILE, STATES_URL);
        int stateRows = processData(STATES_FILE, sTrainingData, sTestingData);

        System.out.println("Total number of county rows: " + countyRows);
        System.out.println("Total number of state rows: " + stateRows);

        // Prepare training and testing data
        StructType schema = new StructType(new StructField[]{
                new StructField("cases", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> cTraining = spark.createDataFrame(cTrainingData, schema);
        Dataset<Row> cTesting = spark.createDataFrame(cTestingData, schema);

        Dataset<Row> sTraining = spark.createDataFrame(sTrainingData, schema);
        Dataset<Row> sTesting = spark.createDataFrame(sTestingData, schema);

        // define the linear regression
        LinearRegression linReg = new LinearRegression().setFeaturesCol("features").setLabelCol("cases");

        // train models
        countyModel = linReg.fit(cTraining);
        stateModel = linReg.fit(sTraining);

        // test models
        testModel(cTesting, countyModel);
        testModel(sTesting, stateModel);
    }

    private static void gatherData(String filename, String url) {
        System.out.println("Generating data from given source: " + url);
        try {
            File dir = new File(DATA_PATH);
            dir.mkdir();

            File file = new File(DATA_PATH + filename);
            FileUtils.copyURLToFile(new URL(url), file);
            System.out.println("Successfully created " + DATA_PATH + filename);
        } catch (MalformedURLException e) {
            System.out.println("Erroneous URL provided.");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("File could not be created.");
            e.printStackTrace();
        }
    }

    private static int processData(String filename, List<Row> trainingData, List<Row> testingData) {
        int rows = 0;
        boolean inCountyMode = filename.contains("counties");
        try (final BufferedReader reader = Files.newBufferedReader(Paths.get(DATA_PATH + filename), StandardCharsets.UTF_8)) {
            Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(reader);
            for (CSVRecord record : records) {
                if(record.get("fips").equals("")) {
                    continue;
                }

                double date = Integer.parseInt(record.get("date").replaceAll("-", ""));
                double fips = Integer.parseInt(record.get("fips"));
                double cases = Integer.parseInt(record.get("cases"));

                String location = (inCountyMode ? record.get("county") + ", " : "" )+ record.get("state");
                fipsToLocation.put(location, fips);

                if(++rows > (inCountyMode ? 18000 : 1300) ) {
                    testingData.add(RowFactory.create(cases, Vectors.dense(fips, date)));
                } else {
                    trainingData.add(RowFactory.create(cases, Vectors.dense(fips, date)));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        return rows;
    }

    private static void testModel(Dataset<Row> testing, LinearRegressionModel model) {
        Dataset<Row> sResults = model.transform(testing);
//        Dataset<Row> sEntries = sResults.select("features", "cases", "prediction");
//        for (Row entry: sEntries.collectAsList()) {
//            System.out.println("(" + entry.get(0) + ", " + entry.get(1) + ") -> prediction=" + entry.get(2));
//        }
    }
    
    private static void makePrediction(String location, String givenDate) {
        // take in a county and a date --> predict cases
        // perhaps give a dropdown to select county,
        // and calendar excluding today and past
        double fips = getFIPS(location);
        double date = Integer.parseInt(givenDate.replaceAll("-", ""));
        Vector input = Vectors.dense(fips, date);

        int casesCt = (int) (location.contains(",") ? countyModel.predict(input) : stateModel.predict(input));
        System.out.println("I predict there will be " + casesCt + " cases in " + location + " on " + givenDate + ".");
    }

    private static double getFIPS(String location) {
        return fipsToLocation.get(location);
    }

    public static void main(String[] args) {
        trainModels();

        makePrediction("Snohomish, Washington", "2020-04-03");
        makePrediction("Los Angeles, California", "2020-04-03");
        makePrediction("Hawaii", "2020-04-03");
        makePrediction("New York", "2020-04-03");

        while (true) {
            System.out.print("");
        }

        // TODO: GUI for makePrediction
    }
}