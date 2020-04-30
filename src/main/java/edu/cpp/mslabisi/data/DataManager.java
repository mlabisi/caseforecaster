package edu.cpp.mslabisi.data;

import org.apache.commons.io.FileUtils;
import org.datavec.api.records.reader.RecordReader;
import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.io.ClassPathResource;

import java.io.File;
import java.io.IOException;
import java.net.MalformedURLException;
import java.net.URL;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

public class DataManager {
    private static final Logger LOG = Logger.getLogger(DataManager.class.getName());

    // create index for mapping names to fips
    private static Map<String, Integer> locationToFIPS = new HashMap<>();
    private static Map<Integer, String> FIPStoLocation = new HashMap<>();

    public static Map<String, Integer> getLocationToFIPS() {
        return locationToFIPS;
    }

    public static Map<Integer, String> getFIPStoLocation() {
        return FIPStoLocation;
    }

    public static void downloadLatest() {
        downloadStates();
        downloadCounties();
    }

    private static void downloadStates() {
        try {
            // store fresh copy of states csv in our resources file
            File infile = new ClassPathResource(Constants.getStatesFilename()).getFile();
            FileUtils.copyURLToFile(new URL(Constants.getStatesUrl()), infile);
        } catch (MalformedURLException e) {
            LOG.severe("‼️ Could not access " + Constants.getStatesUrl() +"\n" + e.getMessage());
        } catch (IOException e) {
            LOG.severe("‼️ Could not create " + Constants.getStatesFilename() +"\n" + e.getMessage());
        }
    }

    private static void downloadCounties() {
        try {
            // store fresh copy of counties csv in our resources file
            File infile = new ClassPathResource(Constants.getCountiesFilename()).getFile();
            FileUtils.copyURLToFile(new URL(Constants.getCountiesUrl()), infile);
        } catch (MalformedURLException e) {
            LOG.severe("‼️ Could not access " + Constants.getCountiesUrl() +"\n" + e.getMessage());
        } catch (IOException e) {
            LOG.severe("‼️ Could not create " + Constants.getCountiesFilename() +"\n" + e.getMessage());
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
            recordReader.initialize(new FileSplit(new ClassPathResource(Constants.getStatesFilename()).getFile()));

            // process the original data and build location to fips map
            while (recordReader.hasNext()) {
                List<Writable> row = recordReader.next();
                if (!row.get(2).toString().equals("")) {
                    String location = row.get(1).toString();
                    int fips = row.get(2).toInt();
                    locationToFIPS.put(location, fips);
                    FIPStoLocation.put(fips, location);
                }
            }
        } catch (InterruptedException | IOException e) {
            LOG.severe("‼️ Could not collect states\n" + e.getMessage());
        }
    }

    private static void collectCounties() {
        try {
            // grab the raw data and convert it to record-like format
            RecordReader recordReader = new CSVRecordReader(1);
            recordReader.initialize(new FileSplit(new ClassPathResource(Constants.getCountiesFilename()).getFile()));

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
