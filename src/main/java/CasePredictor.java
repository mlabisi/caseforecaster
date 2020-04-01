import org.apache.commons.io.FilenameUtils;

import java.util.HashMap;
import java.util.Map;

public class CasePredictor {
    private static Map<String, Double> locationToFIPS = new HashMap<>();

    private static final String COUNTRIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String DATA_PATH = FilenameUtils.concat(System.getProperty("java.io.tmpdir"), "itscoronatime/");
    private static final String COUNTIES_FILE = "covid-19-counties.csv";
    private static final String STATES_FILE = "covid-19-states.csv";

    private static void runModel(String filename, String url) {

    }

    // take in a county and a date --> predict cases
    // perhaps give a dropdown to select county,
    // and calendar excluding today and past
    private static void makePrediction(String location, String date) {
    }

    private static double getFIPS(String location) {
        return locationToFIPS.get(location);
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
        runModel(COUNTIES_FILE, COUNTRIES_URL);
        runModel(STATES_FILE, STATES_URL);
    }

    // TODO: Use a library that doesn't force features to be doubles.
    //       it's important that the features act more like categories/classes
    //       but the total problem should still be a regression problem
    public static void main(String[] args) {
        runModels();

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