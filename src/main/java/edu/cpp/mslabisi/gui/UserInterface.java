package edu.cpp.mslabisi.gui;

import javax.swing.*;
import java.awt.*;

public class UserInterface {
    // frames
    private JFrame welcomeFrame;

    // panels
    private JPanel welcomePanel;

    // display
    private JLabel instruction;

    // buttons
    private JButton beginBtn;
    private JButton confirmLocationBtn;
    private JButton confirmDateBtn;
    private JButton predictAgainBtn;
    private JButton finishBtn;
    private JButton exitBtn;

    public UserInterface() {
        welcomeFrame = new JFrame("Coronavirus Case Predictor");
        welcomePanel = new JPanel();
        instruction = new JLabel("Coronavirus Case Predictor");
    }

    public void initWelcome() {
        welcomePanel.setBorder(BorderFactory.createEmptyBorder(50, 250, 100, 250));
        welcomeFrame.add(welcomePanel, BorderLayout.CENTER);
        beginBtn = new JButton("Begin");
        welcomeFrame.pack();
        welcomeFrame.setVisible(false);
        welcomeFrame.setResizable(false);
        welcomePanel.add(instruction);
        welcomePanel.add(beginBtn);
        welcomeFrame.add(welcomePanel);
    }

    public void showWelcome(boolean show) {
        welcomeFrame.setVisible(show);
    }

    public JButton getBeginBtn() {
        return beginBtn;
    }
}
