package edu.cpp.mslabisi.gui;

import ca.odell.glazedlists.BasicEventList;
import ca.odell.glazedlists.EventList;
import ca.odell.glazedlists.matchers.TextMatcherEditor;
import ca.odell.glazedlists.swing.AutoCompleteSupport;
import ca.odell.glazedlists.swing.DefaultEventComboBoxModel;
import com.github.lgooddatepicker.components.DatePicker;
import com.github.lgooddatepicker.components.DatePickerSettings;
import com.github.lgooddatepicker.zinternaltools.DateVetoPolicyMinimumMaximumDate;
import edu.cpp.mslabisi.data.DataManager;

import javax.swing.*;
import java.awt.*;

public class UserInterface {
    // frames
    private JFrame mainFrame;

    // panels
    private JPanel welcomePanel;
    private JPanel locationPanel;
    private JPanel datePanel;
    private JPanel predictionPanel;
    private JPanel goodbyePanel;

    // outputs
    private JLabel heading;
    private EventList<String> locations;


    // inputs
    private JComboBox<String> locationBox;
    private DatePicker datePicker;

    // buttons
    private JButton beginBtn;
    private JButton locationBtn;
    private JButton dateBtn;
    private JButton predictAgainBtn;
    private JButton finishBtn;
    private JButton exitBtn;

    public UserInterface() {
        mainFrame = new JFrame("Coronavirus Case Predictor");
        mainFrame.setBounds(500, 500, 500, 500);
        mainFrame.setResizable(false);
        mainFrame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

        heading = new JLabel();
        locations = new BasicEventList<>();
        locations.addAll(DataManager.getLocations());

        initWelcome();
        initLocation();
        initDate();
    }

    private void initWelcome() {
        welcomePanel = new JPanel();
        beginBtn = new JButton("Begin");
        welcomePanel.setBorder(BorderFactory.createEmptyBorder(50, 250, 100, 250));
        mainFrame.add(welcomePanel, BorderLayout.CENTER);

        heading.setText("Coronavirus Case Predictor");
        welcomePanel.add(heading);
        welcomePanel.add(beginBtn);
        mainFrame.setVisible(true);
    }

    private void initLocation() {
        locationPanel = new JPanel();
        locationBtn = new JButton("Confirm Location");
        locationPanel.setBorder(BorderFactory.createEmptyBorder(50, 250, 100, 250));

        DefaultEventComboBoxModel<String> model = new DefaultEventComboBoxModel<>(locations);
        locationBox = new JComboBox<>(model);

        SwingUtilities.invokeLater(new Runnable() {
            @Override
            public void run() {
                AutoCompleteSupport autocomplete = AutoCompleteSupport.install(locationBox, locations);
                autocomplete.setFilterMode(TextMatcherEditor.CONTAINS);
            }
        });

        heading.setText("1. Choose a Location");
        locationPanel.add(heading);
        locationPanel.add(locationBox);
        locationPanel.add(locationBtn);
    }

    private void initDate() {
        datePanel = new JPanel();
        dateBtn = new JButton("Confirm Date");
        datePanel.setBorder(BorderFactory.createEmptyBorder(50, 250, 100, 250));

        DatePickerSettings pickerSettings = new DatePickerSettings();
        datePicker = new DatePicker(pickerSettings);
        pickerSettings.setVetoPolicy(new DateVetoPolicyMinimumMaximumDate(DataManager.getMinDate(), DataManager.getMaxDate()));
        datePanel.add(datePicker, getConstraints(1, 16, 1));

        heading.setText("2. Choose a Date");
        datePanel.add(heading);
        datePanel.add(datePicker);
        datePanel.add(dateBtn);
    }

    private void showGoodbye() {
        goodbyePanel = new JPanel();
        exitBtn = new JButton("Exit");
        goodbyePanel.setBorder(BorderFactory.createEmptyBorder(50, 250, 100, 250));

        heading.setText("Wash your hands.\nStay Inside.\nBe safe.\n");
        goodbyePanel.add(heading);
        goodbyePanel.add(exitBtn);
    }

    public void showPrediction(String location, int prediction, String date) {
        predictionPanel = new JPanel();
        predictAgainBtn = new JButton("New Prediction");
        finishBtn = new JButton("Done");
        JPanel buttonsContainer = new JPanel();
        buttonsContainer.setLayout(new FlowLayout(FlowLayout.CENTER));
        buttonsContainer.add(predictAgainBtn);
        buttonsContainer.add(finishBtn);

        heading.setText("I predict " + DataManager.getCaseDifference(prediction) + " new cases in "
                + location + " on " + date + ". This brings the total case count in " + location
                + " to " + prediction + ".");
        predictionPanel.add(heading);
        predictionPanel.add(buttonsContainer);
    }

    public JButton getBeginBtn() {
        return beginBtn;
    }

    public void showLocation() {
        mainFrame.getContentPane().removeAll();
        mainFrame.getContentPane().add(locationPanel, BorderLayout.CENTER);
        mainFrame.revalidate();
        mainFrame.repaint();
    }

    public void showDate() {
        mainFrame.getContentPane().removeAll();
        mainFrame.getContentPane().add(datePanel, BorderLayout.CENTER);
        mainFrame.revalidate();
        mainFrame.repaint();
    }

    public JButton getLocationBtn() {
        return locationBtn;
    }

    public JComboBox<String> getLocationBox() {
        return locationBox;
    }

    public JButton getDateBtn() {
        return dateBtn;
    }

    public DatePicker getDatePicker() {
        return datePicker;
    }

    public JButton getPredictAgainBtn() {
        return predictAgainBtn;
    }

    public JButton getFinishBtn() {
        return finishBtn;
    }

    public JButton getExitBtn() {
        return exitBtn;
    }

    private static GridBagConstraints getConstraints(int gridx, int gridy, int gridwidth) {
        GridBagConstraints gc = new GridBagConstraints();
        gc.fill = GridBagConstraints.NONE;
        gc.anchor = GridBagConstraints.WEST;
        gc.gridx = gridx;
        gc.gridy = gridy;
        gc.gridwidth = gridwidth;
        return gc;
    }
}
