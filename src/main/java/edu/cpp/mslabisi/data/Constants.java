package edu.cpp.mslabisi.data;

public class Constants {
    private static final String COUNTIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILE = "covid_19_counties";
    private static final String STATES_FILE = "covid_19_states";

    public static String getCountiesUrl() {
        return COUNTIES_URL;
    }

    public static String getStatesUrl() {
        return STATES_URL;
    }

    public static String getCountiesFilename() {
        return COUNTIES_FILE;
    }

    public static String getStatesFilename() {
        return STATES_FILE;
    }
}
