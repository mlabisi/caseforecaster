package edu.cpp.mslabisi.plot;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.axis.NumberAxis;
import org.jfree.chart.axis.NumberTickUnit;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.chart.plot.XYPlot;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import javax.swing.*;

public class PlottingTool {
    public static void plot(double[] predicted, double[] actual, int max) {
        double[] index = new double[predicted.length];
        for (int i = 0; i < predicted.length; i++) {
            index[i] = i;
        }

        final XYSeriesCollection dataSet = new XYSeriesCollection();
        addSeries(dataSet, index, predicted, "Predicted");
        addSeries(dataSet, index, actual, "Actual");
        final JFreeChart chart = ChartFactory.createXYLineChart(
                "Expectations vs. Reality", // chart title
                "Index", // x axis label
                "Cases", // y axis label
                dataSet, // data
                PlotOrientation.VERTICAL,
                true, // include legend
                true, // tooltips
                false // urls
        );
        XYPlot xyPlot = chart.getXYPlot();

        // X-axis
        final NumberAxis domainAxis = (NumberAxis) xyPlot.getDomainAxis();
        domainAxis.setRange(index[0], index[predicted.length - 1] + 2);
        domainAxis.setTickUnit(new NumberTickUnit(1));
        domainAxis.setVerticalTickLabels(true);

        // Y-axis
        final NumberAxis rangeAxis = (NumberAxis) xyPlot.getRangeAxis();
        rangeAxis.setRange(0, 1500 * (((double)max / 1500) + 10));
        rangeAxis.setTickUnit(new NumberTickUnit(1500));

        final ChartPanel panel = new ChartPanel(chart);
        final JFrame f = new JFrame();
        f.add(panel);
//        f.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
        f.pack();
        f.setVisible(true);
    }

    private static void addSeries(final XYSeriesCollection dataSet, double[] x, double[] y, final String label) {
        final XYSeries series = new XYSeries(label);
        for (int j = 0; j < x.length; j++) {
            series.add(x[j], y[j]);
        }
        dataSet.addSeries(series);
    }
}
