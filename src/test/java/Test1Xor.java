import org.datavec.api.conf.Configuration;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.junit.Test;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;

//import org.datavec.nlp.vectorizer.TfidfVectorizer;

/**
 * Created by ako on 1/8/2017.
 */
public class Test1Xor {

    @Test
    public void firstTest() {
        info("firstTest");
        Configuration config = new Configuration();
//        TfidfVectorizer vectorizer = new TfidfVectorizer();
//        vectorizer.initialize(config);
//        RecordReader reader = new FileRecordReader();
//        INDArray n = vectorizer.fitTransform(reader);
//        info(String.format("n: %s", n));
   }

    private void info(String msg) {
        System.out.println(msg);
    }

    @Test
    public void simpleNN1() throws InterruptedException {
        info("simpleNN1");

        INDArray input = Nd4j.create(new float[]{0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1}, new int[]{4, 3});
        INDArray labels = Nd4j.create(new float[]{0, 1, 1, 0}, new int[]{4, 1});
        info(input.toString());
        info(labels.toString());

        DataSet ds = new DataSet(input, labels);

        MultiLayerConfiguration config = new NeuralNetConfiguration.Builder()
                .seed(100)
                .iterations(200)
                .learningRate(0.001)
                .useDropConnect(false)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
//                .dist(new UniformDistribution(0, 1))
                .biasInit(0)
                .miniBatch(false)
                .list()
                .layer(0, new DenseLayer.Builder().nIn(3).nOut(6)
                        .activation(Activation.SIGMOID)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
//                        .updater(Updater.SGD)
                        .build())
                .layer(1, new OutputLayer.Builder().nIn(6).nOut(1)
                        .activation(Activation.HARDTANH)
                        .weightInit(WeightInit.DISTRIBUTION)
                        .dist(new UniformDistribution(0, 1))
                        .build())
                .pretrain(false)
                .backprop(true)
                .build();
        MultiLayerNetwork net = new MultiLayerNetwork(config);
        net.init();

        net.setListeners(new ScoreIterationListener(10));

        info(net.output(input).toString());
        net.fit(ds);
        info(net.output(input).toString());

        // show result for train data
        INDArray output = net.output(ds.getFeatureMatrix());
        info(output.toString());

        Evaluation eval = new Evaluation(2);
        eval.eval(ds.getLabels(), output);
        System.out.println(eval.stats());

        //info(net.getLayerWiseConfigurations().toJson());
        //Thread.sleep(1000 * 60 * 10);

        int[] result = net.predict(Nd4j.create(new float[]{1, 0, 1}, new int[]{1, 3}));
        info(Arrays.toString(result));
    }
}
