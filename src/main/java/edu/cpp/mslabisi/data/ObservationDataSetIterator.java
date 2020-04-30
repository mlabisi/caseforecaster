package edu.cpp.mslabisi.data;

import com.opencsv.CSVReader;
import com.opencsv.exceptions.CsvException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.primitives.Pair;

import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.NoSuchElementException;
import java.util.logging.Logger;

public class ObservationDataSetIterator implements DataSetIterator {
    private static final Logger LOG = Logger.getLogger(ObservationDataSetIterator.class.getName());

    // number of features in each time series observation
    private final int FEATURES_CT = 1;

    private int batchSize;
    private int observationsCt; // how many days of data do we want to take in?
    private int predictionCt; // how many days do we want to predict?

    // used to keep track of starting index for each batch, current location pointer
    // will be the head of this linked list
    private LinkedList<Integer> pointerIndices = new LinkedList<>();

    private List<ObservationData> trainingData;
    private List<Pair<INDArray, INDArray>> testingData;

    public ObservationDataSetIterator(String fips, int batchSize, int observationsCt, double splitFactor) {
        List<ObservationData> timeSeries = Integer.parseInt(fips) > 56 ? parseCountyData(fips) : parseStateData(fips);
        this.batchSize = batchSize;
        this.observationsCt = observationsCt;
        this.predictionCt = 1;
        int split = (int) Math.round(timeSeries.size() * splitFactor);
        trainingData = timeSeries.subList(0, split);
        testingData = generateTestingData(timeSeries.subList(split, timeSeries.size()));
        resetPointer();
    }

    @Override
    public DataSet next(int num) {
        if (pointerIndices.size() == 0) {
            throw new NoSuchElementException();
        }

        int practicalBatchSize = Math.min(num, pointerIndices.size());
        INDArray features = Nd4j.create(new int[] {practicalBatchSize, FEATURES_CT, observationsCt}, 'f');
        INDArray label = Nd4j.create(new int[] {practicalBatchSize, predictionCt, observationsCt}, 'f');

        // use each observation in this batch to create a time series
        for(int i = 0; i < practicalBatchSize; i++) {
            int start = pointerIndices.removeFirst();
            int end = start + observationsCt;

            ObservationData thisObs = trainingData.get(start);
            ObservationData nextObs;

            for(int j = start; j < end; j++) {
                int timeStep = j - start;
                features.putScalar(new int[] {i, 0, timeStep}, thisObs.getCases());
                nextObs = trainingData.get(j + 1);
                label.putScalar(new int[] {i, 0, timeStep}, nextObs.getCases());
                thisObs = nextObs;
            }
            if (pointerIndices.size() == 0) {
                break;
            }
        }
        return new DataSet(features, label);
    }

    public List<Pair<INDArray, INDArray>> getTestingData() {
        return testingData;
    }

    private List<ObservationData> parseCountyData(String targetFips) {
        List<ObservationData> observationData = new ArrayList<>();
        try {
            List<String[]> rawData = new CSVReader(new FileReader(Constants.getCountiesRsc())).readAll();
            observationData = collectRows(targetFips, rawData);
        } catch (IOException | CsvException e) {
            LOG.severe("‼️ Could not read " + Constants.getCountiesRsc() + "\n" + e.getMessage());
        }
        return observationData;
    }

    private List<ObservationData> parseStateData(String targetFips) {
        List<ObservationData> observationData = new ArrayList<>();
        try {
            List<String[]> rawData = new CSVReader(new FileReader(Constants.getStatesRsc())).readAll();
            observationData = collectRows(targetFips, rawData);
        } catch (IOException | CsvException e) {
            LOG.severe("‼️ Could not read " + Constants.getStatesRsc() + "\n" + e.getMessage());
        }
        return observationData;
    }

    private List<ObservationData> collectRows(String targetFips, List<String[]> rawData) {
        List<ObservationData> observationData = new ArrayList<>();
        for (String[] row : rawData) {
            String fips;
            if (!(fips = row[row.length - 3]).equals(targetFips)) {
                continue;
            }

            String date = row[0];
            String location = DataManager.getLocationFromFips(Integer.parseInt(fips));
            int cases = Integer.parseInt(row[row.length - 2]);

            observationData.add(new ObservationData(date, location, fips, cases));
        }
        return observationData;
    }

    private List<Pair<INDArray, INDArray>> generateTestingData(List<ObservationData> observationData) {
        int n = observationsCt + predictionCt;
        List<Pair<INDArray, INDArray>> testingData = new ArrayList<>();

        // for each observation in a window of size n = observationCt, use the features to predict the
        // next observation's label (aka, the features of the label we're trying to predict are each of
        // the n previous observations.
        for (int i = 0; i < observationData.size() - n; i++) {
            INDArray features = Nd4j.create(new int[]{observationsCt, FEATURES_CT}, 'f');
            for (int j = i; j < i + observationsCt; j++) {
                ObservationData observation = observationData.get(j);
                features.putScalar(new int[] {j - i, 0}, observation.getCases());
            }
            ObservationData observation = observationData.get(i + observationsCt);
            INDArray label = Nd4j.create(new int[] {1}, 'f');
            label.putScalar(new int[]{0}, observation.getCases());
            testingData.add(new Pair<>(features, label));
        }
        return testingData;
    }

    private void resetPointer() {
        pointerIndices.clear();
        int window = observationsCt + predictionCt;
        for(int i = 0; i < trainingData.size() - window; i++) {
            pointerIndices.add(i);
        }
    }

    @Override
    public int inputColumns() {
        return FEATURES_CT;
    }

    @Override
    public int totalOutcomes() {
        return predictionCt;
    }

    @Override
    public boolean resetSupported() {
        return false;
    }

    @Override
    public boolean asyncSupported() {
        return false;
    }

    @Override
    public void reset() {
        resetPointer();
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public DataSetPreProcessor getPreProcessor() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public List<String> getLabels() {
        throw new UnsupportedOperationException("Not Implemented");
    }

    @Override
    public boolean hasNext() {
        return pointerIndices.size() > 0;
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }
}
