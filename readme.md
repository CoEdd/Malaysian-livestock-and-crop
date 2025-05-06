# Malaysian Livestock and Crop Production Dashboard

## Overview
The **Malaysian Livestock and Crop Production Dashboard** is an interactive web application designed to analyze and forecast agricultural commodity production trends in Malaysia. This project integrates **data visualization**, **machine learning models**, and **time series forecasting** to provide actionable insights for stakeholders in the agricultural sector.

The dashboard combines **Power BI visualizations**, **KMeans clustering**, and **ARIMA forecasting** to deliver a comprehensive tool for understanding historical trends, identifying patterns, and predicting future production.

---

## Features
### 1. **Interactive Dashboard**
- **Power BI Integration**: Embedded Power BI dashboard for visualizing production trends of livestock and crops.
- **Commodity Statistics**: Key metrics such as total commodities tracked, data coverage, and forecast horizon.

### 2. **Machine Learning Models**
- **KMeans Clustering**: Groups commodities based on production growth rates and volatility to identify similar patterns.
- **ARIMA Forecasting**: Predicts future production trends for selected commodities with metrics like R², MAE, and RMSE.

### 3. **User-Friendly Interface**
- **Dynamic Tabs**: Switch between Power BI visualizations and machine learning models seamlessly.
- **Interactive Controls**: Select commodities and run models with a single click.

---

## Technologies Used
### **Frontend**
- **HTML5**, **CSS3**, **JavaScript**
- **Tailwind CSS** for responsive design
- **jQuery** for AJAX requests
- **Font Awesome** for icons

### **Backend**
- **Flask**: Lightweight Python web framework for API development
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib**: Data visualization
- **Scikit-learn**: KMeans clustering
- **Statsmodels**: ARIMA time series forecasting
- **Pillow**: Image processing

### **Data Visualization**
- **Power BI**: Embedded dashboard for interactive visualizations

---

## Key Skills Demonstrated
- **Data Analysis**: Cleaning, preprocessing, and analyzing agricultural production data.
- **Machine Learning**: Implemented KMeans clustering to identify patterns in commodity production.
- **Time Series Forecasting**: Built ARIMA models to predict future production trends.
- **Web Development**: Developed a full-stack web application with Flask and integrated Power BI.
- **Data Visualization**: Created interactive visualizations using Matplotlib and Power BI.
- **API Development**: Designed RESTful APIs for serving machine learning results and forecasts.

---

## How It Works
1. **Data Loading**: The application reads data from an Excel file containing historical production data for various commodities.
2. **KMeans Clustering**:
   - Calculates average production growth rates and volatility for each commodity.
   - Groups commodities into clusters based on their characteristics.
   - Displays the results in a scatter plot with annotations.
3. **ARIMA Forecasting**:
   - Builds ARIMA models for selected commodities.
   - Forecasts production for the next 10 years.
   - Displays historical data, forecasts, and model performance metrics.
4. **Power BI Dashboard**:
   - Provides an overview of production trends and key statistics.

---

