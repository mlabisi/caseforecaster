package edu.cpp.mslabisi.data;

import org.nd4j.linalg.io.ClassPathResource;

public class Constants {
    private static final String COUNTIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILENAME = "covid_19_counties";
    private static final String STATES_FILENAME = "covid_19_states";
    private static final String MODEL_FILENAME = "model.rnn";
    private static final ClassPathResource COUNTIES_RSC = new ClassPathResource(Constants.getCountiesFilename());
    private static final ClassPathResource STATES_RSC = new ClassPathResource(Constants.getStatesFilename());
    private static final ClassPathResource MODEL_RSC = new ClassPathResource(Constants.getModelFilename());

    public static String getCountiesUrl() {
        return COUNTIES_URL;
    }

    public static String getStatesUrl() {
        return STATES_URL;
    }

    public static String getCountiesFilename() {
        return COUNTIES_FILENAME;
    }

    public static String getStatesFilename() {
        return STATES_FILENAME;
    }

    public static String getModelFilename() {
        return MODEL_FILENAME;
    }

    public static ClassPathResource getCountiesRsc() {
        return COUNTIES_RSC;
    }

    public static ClassPathResource getStatesRsc() {
        return STATES_RSC;
    }

    public static ClassPathResource getModelRsc() {
        return MODEL_RSC;
    }
}
