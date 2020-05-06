package edu.cpp.mslabisi.data;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;

import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.time.LocalDate;
import java.time.Period;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.logging.Logger;

public class DataManager {
    private static final Logger LOG = Logger.getLogger(DataManager.class.getName());

    // create index for mapping names to fips
    private static Map<String, Integer> locationToFIPS = new HashMap<>();
    private static Map<Integer, String> FIPStoLocation = new HashMap<>();

    private static LocalDate minDate;
    private static LocalDate maxDate;

    public static void initialize() {
        Constants.getRscDir().mkdir();
        downloadLatest();
        processData();
    }

    public static Set<String> getLocations() {
        return locationToFIPS.keySet();
    }

    public static LocalDate getMinDate() {
        return minDate;
    }

    public static LocalDate getMaxDate() {
        return maxDate;
    }

    public static int getFipsFromLocation(String location) {
        if (locationToFIPS.containsKey(location)) {
            return locationToFIPS.get(location);
        } else {
            LOG.severe("‼️ Could not find " + location + " in database");
        }

        return -1;
    }

    public static int getDaysDifference(String targetDate) {
        LocalDate target = LocalDate.parse(targetDate);
        Period difference = Period.between(minDate, target);
        return difference.getDays();
    }

    public static String getLocationFromFips(int fips) {
        if (FIPStoLocation.containsKey(fips)) {
            return FIPStoLocation.get(fips);
        } else {
            LOG.severe("‼️ Could not find fips \"" + fips + "\" in database");
        }

        return "";
    }

    public static void downloadLatest() {
        downloadStates();
        downloadCounties();
    }

    private static void downloadStates() {
        try {
            // store fresh copy of states csv in our resources directory
            FileUtils.copyURLToFile(new URL(Constants.getStatesUrl()), Constants.getStatesRsc());
        } catch (MalformedURLException e) {
            LOG.severe("‼️ Could not access " + Constants.getStatesUrl() + "\n" + e.getMessage());
        } catch (IOException e) {
            LOG.severe("‼️ Could not create " + Constants.getStatesFilename() + "\n" + e.getMessage());
        }
    }

    private static void downloadCounties() {
        try {
            // store fresh copy of counties csv in our resources directory
            FileUtils.copyURLToFile(new URL(Constants.getCountiesUrl()), Constants.getCountiesRsc());
        } catch (MalformedURLException e) {
            LOG.severe("‼️ Could not access " + Constants.getCountiesUrl() + "\n" + e.getMessage());
        } catch (IOException e) {
            LOG.severe("‼️ Could not create " + Constants.getCountiesFilename() + "\n" + e.getMessage());
        }
    }

    public static void processData() {
        collectStates();
        collectCounties();
    }

    private static void collectStates() {
        try {
            // grab the raw data and convert it to record-like format
            RecordReader recordReader = new CSVRecordReader(1);
            recordReader.initialize(new FileSplit(Constants.getStatesRsc()));

            // process the original data and build location to fips map
            while (recordReader.hasNext()) {
                List<Writable> row = recordReader.next();
                if (!row.get(2).toString().equals("")) {
                    // store the most recent date for time-series step calculation
                    String lastDate = row.get(0).toString();
                    minDate = LocalDate.parse(lastDate);
                    String location = row.get(1).toString();
                    int fips = row.get(2).toInt();
                    locationToFIPS.put(location, fips);
                    FIPStoLocation.put(fips, location);
                }
            }

            maxDate = minDate.plusDays(21);
        } catch (InterruptedException | IOException e) {
            LOG.severe("‼️ Could not collect states\n" + e.getMessage());
        }
    }

    private static void collectCounties() {
        try {
            // grab the raw data and convert it to record-like format
            RecordReader recordReader = new CSVRecordReader(1);
            recordReader.initialize(new FileSplit(Constants.getCountiesRsc()));

            // process the original data and build location to fips map
            while (recordReader.hasNext()) {
                List<Writable> row = recordReader.next();
                if (!row.get(3).toString().equals("")) {
                    String location = row.get(1).toString() + ", " + row.get(2).toString();
                    int fips = row.get(3).toInt();
                    locationToFIPS.put(location, fips);
                    FIPStoLocation.put(fips, location);
                }
            }
        } catch (InterruptedException | IOException e) {
            LOG.severe("‼️ Could not collect counties\n" + e.getMessage());
        }
    }
}
