package edu.cpp.mslabisi.predict;

import edu.cpp.mslabisi.data.DataManager;
import edu.cpp.mslabisi.data.ObservationDataSetIterator;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
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
            System.out.print("Please enter a location:\n> ");
            String location = in.nextLine();
            System.out.print("Please enter a date:\n> ");
            String date = in.nextLine();

            makePrediction(location, date);
            System.out.print("Would you like to make another prediction?\n> ");
            answer = in.nextLine();
        } while (answer.startsWith("y") || answer.startsWith("Y"));
        System.exit(0);
    }

    private static void makePrediction(String location, String date) {
        // take in a county and a date --> predict cases
        // perhaps give a dropdown to select county,
        // and calendar excluding today and past

        int fips = DataManager.getFipsFromLocation(location);
        int timeSteps = DataManager.getDaysDifference(date);

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
        }

        LOG.info("📥 Saving updated LSTM model");
        RNN.saveModel(model);

        LOG.info("📤 Restoring newly updated LSTM model");
        model = RNN.restoreModel();

        LOG.info("⚙️ Testing LSTM model");
        int max = iterator.getMaxArray()[0];
        int min = iterator.getMinArray()[0];
        int prediction = predict(model, testingData, timeSteps, max, min);
        System.out.println("I predict there will be " + prediction + " cases in " + location + " on " + date + ".");
    }

    private static int predict(ComputationGraph model, List<Pair<INDArray, INDArray>> testingData, int timeSteps, int max, int min) {
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

//        LOG.info("📈 Now plotting results");
//        PlottingTool.plot(predictions, actuals, max);

        return (int) Math.round(predict(model, testingData.get(testingData.size() - 1).getKey(), testingData.get(testingData.size() - 1).getValue().getDouble(0), timeSteps, max, min));
    }

    private static double predict(ComputationGraph model, INDArray features, double label, int timeSteps, int max, int min) {
        INDArray inputs = Nd4j.create(features.rows(), 1);
        for (int i = 0; i < features.rows() - 1; i++) {
            inputs.putScalar(i, 0, features.getDouble(i + 1, 0));
        }
        inputs.putScalar(features.rows() - 1, 0, (label - min) / (max - min));
        label = model.rnnTimeStep(inputs)[0].getDouble(timeSteps) * (max - min) + min;

        return label;
    }
}
