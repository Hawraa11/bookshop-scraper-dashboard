# bookshop-scraper-dashboard
A Python project that scrapes book data from multiple sources — including Lebanese bookstores, Goodreads, and various APIs — and provides an interactive dashboard for data visualization and analysis. This tool helps users explore book prices, trends, and insights across different platforms in a unified, easy-to-use interface.

# 📚 Booky: Smart Book Scraper & Trend Dashboard

Booky is an integrated platform that collects, analyzes, and visualizes book data from local Lebanese bookstores, global platforms like Goodreads, and external APIs such as OMDb and Google Books. Built with Python and Streamlit, this project offers a comprehensive, interactive dashboard for exploring book availability, pricing, genres, reviews, and movie adaptations.

---

## 🚀 Project Overview

This project goes beyond basic web scraping by merging data from multiple sources:

- *Local Lebanese Bookstores:* Antoine Online, Koala Shop, and The Circle Bookshop  
- *Global Platform:* Goodreads datasets sourced from Hugging Face  
- *External APIs:* OMDb API for movie adaptations and Google Books API for enriched book metadata

The combined data powers an intuitive Streamlit dashboard that helps users explore, compare, and discover books across local and global markets.

---

## 📦 Data Collection & Processing

- Scraped *22,073 books* from Lebanese bookstores using custom Python scripts with BeautifulSoup and Selenium.  
- Collected Goodreads data by merging two datasets from Hugging Face: one with 2,015 books scraped live, and another existing dataset of 10,001 books.  
- Applied data cleaning and merging to unify datasets for consistent analysis.  
- Implemented parallel scraping techniques to speed up data collection, including opening multiple browser sessions.  
- Developed robust error handling for API requests and web scraping challenges.

---

## 🎯 Key Features

- *Multi-source scraping:* Handles different HTML structures for each bookstore with separate scraping scripts.  
- *Price Distribution & Book Count:* Visualizations showing price ranges and inventory sizes per bookstore.  
- *Genre Analysis:* Distribution of book genres by store and across all shops.  
- *Book Title Word Cloud:* Visual summary of common themes in Lebanese book titles.  
- *Book-to-Movie Adaptation Checker:* Integration with OMDb API to display movie info for books.  
- *Rich Book Metadata:* Integration with Google Books API for additional book details.  
- *Interactive Dashboard:* Built with Streamlit for easy web-based exploration and visualization.

---

## 🛠 Technologies Used

- Python  
- BeautifulSoup  
- Selenium  
- Pandas  
- Streamlit  
- OMDb API  
- Google Books API  
- Hugging Face Datasets  
- Matplotlib & WordCloud

---

## 💻 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/Hawraa11
