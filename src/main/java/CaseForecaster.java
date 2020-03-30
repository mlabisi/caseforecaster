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
import java.util.List;

public class CaseForecaster {
    private static LinearRegressionModel model;

    private static void gatherData(String dataURL) {
        String dataPath = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
        String fileName = "covid-19-counties.csv";

        SparkSession spark = SparkSession.builder().master("local[*]").appName("Case Predictor").getOrCreate();

        List<Row> trainingData = new ArrayList<>();
        List<Row> testingData = new ArrayList<>();
        int rows = 0;

        System.out.println("Generating data from given source: " + dataURL);
        try {
            File dir = new File(dataPath);
            dir.mkdir();

            File file = new File(dataPath + fileName);
            FileUtils.copyURLToFile(new URL(dataURL), file);
            System.out.println("Successfully created " + dataPath + fileName);
        } catch (MalformedURLException e) {
            System.out.println("Erroneous URL provided.");
            e.printStackTrace();
        } catch (IOException e) {
            System.out.println("File could not be created.");
            e.printStackTrace();
        }

        try (final BufferedReader reader = Files.newBufferedReader(Paths.get(dataPath + fileName), StandardCharsets.UTF_8)) {
            Iterable<CSVRecord> records = CSVFormat.RFC4180.withFirstRecordAsHeader().parse(reader);
            for (CSVRecord record : records) {
                if(record.get("fips").equals("")) {
                    continue;
                }

                double date = Integer.parseInt(record.get("date").replaceAll("-", ""));
                double fips = Integer.parseInt(record.get("fips"));
                double cases = Integer.parseInt(record.get("cases"));

                if(++rows > 10000 ) {
                    testingData.add(RowFactory.create(cases, Vectors.dense(fips, date)));
                } else {
                    trainingData.add(RowFactory.create(cases, Vectors.dense(fips, date)));
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        // Prepare training and testing data
        StructType schema = new StructType(new StructField[]{
                new StructField("cases", DataTypes.DoubleType, false, Metadata.empty()),
                new StructField("features", new VectorUDT(), false, Metadata.empty())
        });
        Dataset<Row> training = spark.createDataFrame(trainingData, schema);
        Dataset<Row> testing = spark.createDataFrame(testingData, schema);

        // define the linear regression
        // identify features and label
        LinearRegression linReg = new LinearRegression().setFeaturesCol("features").setLabelCol("cases");

        // train model
        model = linReg.fit(training);

        // test model
        Dataset<Row> results = model.transform(testing);
//        Dataset<Row> entries = results.select("features", "cases", "prediction");
//        for (Row entry: entries.collectAsList()) {
//            System.out.println("(" + entry.get(0) + ", " + entry.get(1) + ") -> prediction=" + entry.get(2));
//        }
    }

    private static void makePrediction(String county, String date) {
        // take in a county and a date --> predict cases
        // perhaps give a dropdown to select county,
        // and calendar excluding today and past
        double fips = getCounty(county);
        double datecode = Integer.parseInt(date.replaceAll("-", ""));

        Vector input = Vectors.dense(fips, datecode);
        int casesCt = (int) model.predict(input);
        System.out.println("I predict there will be " + casesCt + " cases in " + county + " on " + date + ".");
    }

    private static double getCounty(String county) {
        return 53061.0;
    }

    public static void main(String[] args) {
        gatherData("https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv");
        makePrediction("Snohomish, WA", "2020-05-24");
    }
}
