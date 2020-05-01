package edu.cpp.mslabisi.predict;

import edu.cpp.mslabisi.data.DataManager;
import edu.cpp.mslabisi.data.ObservationDataSetIterator;
import edu.cpp.mslabisi.plot.PlottingTool;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;
import java.util.Scanner;
import java.util.logging.Logger;

public class CaseCountPredictor {
    private static final Logger LOG = Logger.getLogger(CaseCountPredictor.class.getName());

    private static int observationsCt = 21; // # of observations used to predict next day's case ct
    private static int batchSize = 50; // # of observations to be handled at a time
    private static double splitFactor = 0.5; // aka % of observations to be used for training
    private static int epochs = 111; // training epochs

    public static void main(String[] args) {
        setup();
        userLoop();
    }

    private static void setup() {
        DataManager.initialize();
    }

    private static void userLoop() {
        Scanner in = new Scanner(System.in);
        String answer;
        do {
            System.out.print("Please enter a location: ");
            String location = in.nextLine();
            System.out.print("Please enter a date: ");
            String date = in.nextLine();

            makePrediction(location, date);
            System.out.print("Would you like to make another prediction? (Y/N): ");
            answer = in.nextLine();
        } while (answer.startsWith("y") || answer.startsWith("Y"));
        System.exit(0);
    }

    private static void makePrediction(String location, String date) {
        // take in a county and a date --> predict cases
        // perhaps give a dropdown to select county,
        // and calendar excluding today and past

        // date doesn't do anything as of now
        // later, incorporate date to allow for guess on specific date

        int fips = DataManager.getFipsFromLocation(location);

        LOG.info("🔨 Creating DataSetIterator");
        ObservationDataSetIterator iterator = new ObservationDataSetIterator(
                String.valueOf(fips),
                batchSize,
                observationsCt,
                splitFactor);

        LOG.info("🗂 Loading testing data");
        List<Pair<INDArray, INDArray>> testingData = iterator.getTestingData();

        LOG.info("📤 Loading LSTM model");
        ComputationGraph model = RNN.getModel();

        LOG.info("⚙️ Training LSTM model");
        for (int i = 0; i < epochs; i++) {
            while (iterator.hasNext()) {
                model.fit(iterator.next());
            }
            iterator.reset();
            model.rnnClearPreviousState();
        }

        LOG.info("📥 Saving updated LSTM model");
        RNN.saveModel(model);

        LOG.info("📤 Restoring newly updated LSTM model");
        model = RNN.restoreModel();

        LOG.info("⚙️ Testing LSTM model");
        int max = iterator.getMaxArray()[0];
        int min = iterator.getMinArray()[0];
        predictAndPlot(model, testingData, max, min);

    }

    private static void predictAndPlot(ComputationGraph model, List<Pair<INDArray, INDArray>> testingData, int max, int min) {
        double[] predictions = new double[testingData.size()];
        double[] actuals = new double[testingData.size()];


        for (int i = 0; i < testingData.size(); i++) {
            predictions[i] = model.rnnTimeStep(testingData.get(i).getKey())[0].getDouble(observationsCt - 1) * (max - min) + min;
            actuals[i] = testingData.get(i).getValue().getDouble(0);
        }
        LOG.info("🔍 Expectations vs Reality for LSTM model");
        for (int i = 0; i < predictions.length; i++) {
            LOG.info(actuals[i] + ", " + predictions[i]);
        }

        LOG.info("📈 Now plotting results");
        PlottingTool.plot(predictions, actuals);
    }
}
