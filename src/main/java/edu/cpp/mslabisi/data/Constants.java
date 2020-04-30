package edu.cpp.mslabisi.data;

import org.apache.commons.io.FilenameUtils;

import java.io.File;

public class Constants {
    private static final String COUNTIES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv";
    private static final String STATES_URL = "https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-states.csv";
    private static final String COUNTIES_FILENAME = "covid_19_counties.csv";
    private static final String STATES_FILENAME = "covid_19_states.csv";
    private static final String MODEL_FILENAME = "model.zip";

    private static final File RSC_DIR = new File(FilenameUtils.concat(System.getProperty("user.dir") + "/src/main", "resources"));
    private static final File COUNTIES_RSC = new File(FilenameUtils.concat(RSC_DIR.getPath(), getCountiesFilename()));
    private static final File STATES_RSC = new File(FilenameUtils.concat(RSC_DIR.getPath(), getStatesFilename()));
    private static final File MODEL_RSC = new File(FilenameUtils.concat(RSC_DIR.getPath(), getModelFilename()));

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

    public static File getRscDir() {
        return RSC_DIR;
    }

    public static File getCountiesRsc() {
        return COUNTIES_RSC;
    }

    public static File getStatesRsc() {
        return STATES_RSC;
    }

    public static File getModelRsc() {
        return MODEL_RSC;
    }
}
