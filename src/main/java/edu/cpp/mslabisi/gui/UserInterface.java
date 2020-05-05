package edu.cpp.mslabisi.gui;

import ca.odell.glazedlists.BasicEventList;
import ca.odell.glazedlists.EventList;
import ca.odell.glazedlists.matchers.TextMatcherEditor;
import ca.odell.glazedlists.swing.AutoCompleteSupport;
import ca.odell.glazedlists.swing.DefaultEventComboBoxModel;
import edu.cpp.mslabisi.data.DataManager;

import javax.swing.*;
import java.awt.*;

public class UserInterface {
    // frames
    private JFrame mainFrame;

    // panels
    private JPanel welcomePanel;
    private JPanel locationPanel;

    // outputs
    private JLabel instruction;
    private EventList<String> locations;


    // inputs
    private JComboBox<String> locationBox;

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

        instruction = new JLabel();
        locations = new BasicEventList<>();
        locations.addAll(DataManager.getLocations());

        initWelcome();
        initLocation();
    }

    private void initWelcome() {
        welcomePanel = new JPanel();
        beginBtn = new JButton("Begin");
        welcomePanel.setBorder(BorderFactory.createEmptyBorder(50, 250, 100, 250));
        mainFrame.add(welcomePanel, BorderLayout.CENTER);

        instruction.setText("Coronavirus Case Predictor");
        welcomePanel.add(instruction);
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

        instruction.setText("1. Choose a Location");
        locationPanel.add(instruction);
        locationPanel.add(locationBox);
        locationPanel.add(locationBtn);
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

    public JButton getLocationBtn() {
        return locationBtn;
    }
}
