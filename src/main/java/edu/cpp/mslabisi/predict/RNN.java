package edu.cpp.mslabisi.predict;

import edu.cpp.mslabisi.data.Constants;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.ComputationGraphConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.LSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.graph.ComputationGraph;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.AMSGrad;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.IOException;
import java.util.logging.Logger;

public class RNN {
    private static final Logger LOG = Logger.getLogger(RNN.class.getName());

    private static final int seed = 54321;
    private static final int layer1Size = 1; // 156; not working due to mismatched shape :/
    private static final double learningRate = 0.005;
    private static final int nIn = 1;
    private static final int nOut = 1;

    public static ComputationGraph getModel() {
        ComputationGraph model = null;
        if (!Constants.getModelRsc().exists()) {
            // configure model
            ComputationGraphConfiguration conf = new NeuralNetConfiguration.Builder()
                    .seed(seed)
                    .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                    .weightInit(WeightInit.XAVIER)
                    .updater(new AMSGrad(learningRate))
                    .graphBuilder()
                    .addInputs("features")
                    .setOutputs("caseCt")
                    .addLayer("L1", new LSTM.Builder()
                                    .nIn(nIn)
                                    .nOut(layer1Size)
                                    .forgetGateBiasInit(1)
                                    .activation(Activation.TANH)
                                    .build(),
                            "features")
                    .addLayer("caseCt", new RnnOutputLayer.Builder(LossFunctions.LossFunction.MSE)
                            .activation(Activation.IDENTITY)
                            .nIn(layer1Size).nOut(nOut).build(),"L1")
                    .build();

            // build model
            model = new ComputationGraph(conf);
            model.init();
            model.setListeners(new ScoreIterationListener(10));
            saveModel(model);
        } else {
            model = restoreModel();
        }

        return model;
    }

    public static void saveModel(ComputationGraph model) {
        try {
            model.save(Constants.getModelRsc(), true);
        } catch (IOException e) {
            LOG.severe("‼️ Could not save LSTM model\n" + e.getMessage());
        }
    }

    public static ComputationGraph restoreModel() {
        try {
            return ComputationGraph.load(Constants.getModelRsc(), true);
        } catch (IOException e) {
            LOG.severe("‼️ Could not restore LSTM model\n" + e.getMessage());
        }
        return null;
    }
}
