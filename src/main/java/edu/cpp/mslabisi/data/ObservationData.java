package edu.cpp.mslabisi.data;

public class ObservationData {
    private String date;
    private String location;
    private String fips;
    private double cases;

    public ObservationData(String date, String location, String fips, int cases) {
        this.date = date;
        this.location = location;
        this.fips = fips;
        this.cases = cases;
    }

    public String getDate() {
        return date;
    }

    public void setDate(String date) {
        this.date = date;
    }

    public String getLocation() {
        return location;
    }

    public void setLocation(String location) {
        this.location = location;
    }

    public String getFips() {
        return fips;
    }

    public void setFips(String fips) {
        this.fips = fips;
    }

    public double getCases() {
        return cases;
    }

    public void setCases(double cases) {
        this.cases = cases;
    }
}
