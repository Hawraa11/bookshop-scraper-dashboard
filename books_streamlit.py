import streamlit as st
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import openai
import io
from wordcloud import WordCloud
import streamlit as st
import pandas as pd
import numpy as np
import json
import os # Import os if needed for path handling (though r"" makes it less critical here)
import re 



# --- Data Loading and Preprocessing ---

# Use caching to avoid reloading data every time the app interacts
@st.cache_data
def load_goodreads_data(file1_path, file2_path):
    """Loads and preprocesses data from two Goodreads CSV files."""
    df1 = pd.DataFrame()
    df2 = pd.DataFrame()
    load_errors = []

    # --- Load File 1 ---
    try:
        # Assuming comma delimiter and UTF-8 encoding
        # Try common encodings if UTF-8 fails: 'latin1', 'cp1252', 'ISO-8859-1'
        # If tab-separated: sep='\t'
        df1 = pd.read_csv(file1_path, encoding='utf-8')
        # Rename columns for consistency based on your screenshots
        df1 = df1.rename(columns={
            'Avg Rating': 'Avg Rating',
            'Rating Count': 'Rating Count',
            'Review Count': 'Review Count',
            'Cover Image': 'Cover Image URL', # Consistent naming
            'Genre': 'Genre_File1' # Differentiate if needed
            # Assuming 'Title' and 'Author' are already correct in df1 based on screenshot
        })
        # --- Robust Cleaning for Rating/Count columns in df1 ---
        numeric_cols_df1 = ['Avg Rating', 'Rating Count', 'Review Count']
        for col in numeric_cols_df1:
            if col in df1.columns:
                # Clean potentially problematic characters like commas before converting to numeric
                # Use regex to remove anything that is NOT a digit or a dot
                if df1[col].dtype == 'object': # Only attempt if the column is read as object (string)
                    df1[col] = df1[col].astype(str).str.replace('[^0-9.]', '', regex=True)
                df1[col] = pd.to_numeric(df1[col], errors='coerce') # Convert to numeric, turn errors into NaN


    except FileNotFoundError:
        load_errors.append(f"Error: {file1_path} not found.")
        df1 = pd.DataFrame() # Ensure df1 is empty if file not found
    except Exception as e:
        load_errors.append(f"Error loading {file1_path}: {e}")
        df1 = pd.DataFrame()


    # --- Load File 2 ---
    try:
        # Assuming comma delimiter and UTF-8 encoding
        df2 = pd.read_csv(file2_path, encoding='utf-8')
        # Rename columns for consistency based on your screenshots
        df2 = df2.rename(columns={
            'Book': 'Title', # 'Book' column in file 2 seems like 'Title' in file 1
            'Avg Rating': 'Avg Rating', # Same name
            'Rating Count': 'Rating Count', # Same name
            'Review Count': 'Review Count', # Added based on possibility it's in df2 too
            # Assuming 'Author' and 'Description' are correct in df2
            'Link': 'Goodreads Link', # More descriptive name
            'J': 'Cover Image URL_File2', # Using column letter as temporary name, check if it's the cover image
            'L': 'Tag_Column1', # Using column letter, need to understand content
            'M': 'Tag_Column2'  # Using column letter, need to understand content
            # Assuming 'H', 'I', 'K', 'N', 'O', 'Q' are potential numeric columns
        })
        # --- Robust Cleaning for Rating/Count columns in df2 ---
        numeric_cols_df2 = ['Avg Rating', 'Rating Count', 'Review Count']
        for col in numeric_cols_df2:
            if col in df2.columns:
                # Clean potentially problematic characters like commas before converting to numeric
                if df2[col].dtype == 'object': # Only attempt if the column is read as object (string)
                    df2[col] = df2[col].astype(str).str.replace('[^0-9.]', '', regex=True)
                df2[col] = pd.to_numeric(df2[col], errors='coerce') # Convert to numeric, turn errors into NaN


        # Convert potential other numeric columns identified by letter to numeric
        other_numeric_cols_potential = ['H', 'I', 'K', 'N', 'O', 'Q'] # Based on both screenshots
        for col in other_numeric_cols_potential:
            if col in df2.columns:
                df2[col] = pd.to_numeric(df2[col], errors='coerce')


    except FileNotFoundError:
        load_errors.append(f"Error: {file2_path} not found.")
        df2 = pd.DataFrame() # Ensure df2 is empty if file not found
    except Exception as e:
        load_errors.append(f"Error loading {file2_path}: {e}")
        df2 = pd.DataFrame()

    # Display any loading errors
    for error in load_errors:
        st.error(error)


    # --- Combine DataFrames ---
    combined_df = pd.DataFrame()

    if not df1.empty or not df2.empty: # Proceed if at least one DF loaded
        merge_cols = ['Title', 'Author']
        # Check which merge columns are present and not all NaN in each dataframe
        merge_cols_df1_present_and_not_empty = [col for col in merge_cols if col in df1.columns and df1[col].dropna().empty == False]
        merge_cols_df2_present_and_not_empty = [col for col in merge_cols if col in df2.columns and df2[col].dropna().empty == False]


        # Check if there are common, non-empty columns to merge on
        if merge_cols_df1_present_and_not_empty and merge_cols_df1_present_and_not_empty == merge_cols_df2_present_and_not_empty:
            try:
                # Perform the merge based on common merge columns
                combined_df = pd.merge(
                    df1,
                    df2,
                    on=merge_cols_df1_present_and_not_empty,
                    how='outer', # Use 'outer' to keep all rows from both DFs
                    suffixes=('_file1', '_file2') # Add suffixes to overlapping columns not in 'on'
                )

                # --- Data Cleaning and Consolidation after merge ---
                # Consolidate overlapping columns, prioritizing non-null values
                cols_to_consolidate = ['Avg Rating', 'Rating Count', 'Review Count', 'Description', 'Cover Image URL']
                for col in cols_to_consolidate:
                    col1_name = f'{col}_file1'
                    col2_name = f'{col}_file2'

                    if col1_name in combined_df.columns and col2_name in combined_df.columns:
                        # Use coalesce-like logic: take value from file1 if not null, else take from file2
                        combined_df[col] = combined_df[col1_name].fillna(combined_df[col2_name])
                        # Drop the original suffixed columns
                        combined_df = combined_df.drop(columns=[col1_name, col2_name])
                    elif col1_name in combined_df.columns:
                        combined_df[col] = combined_df[col1_name]
                        combined_df = combined_df.drop(columns=[col1_name])
                    elif col2_name in combined_df.columns:
                        combined_df[col] = combined_df[col2_name]
                        combined_df = combined_df.drop(columns=[col2_name])
                    # If the column existed before merge and wasn't duplicated, it keeps its original name


                # --- Consolidate Genre/Tags ---
                # Prioritize 'Genre_File1', then look at 'Tag_Column1', 'Tag_Column2', 'L', 'M'
                genre_column_options_after_merge = ['Genre_File1', 'Tag_Column1', 'Tag_Column2', 'L', 'M']
                genre_column_found = None

                for col in genre_column_options_after_merge:
                    if col in combined_df.columns and combined_df[col].dropna().empty == False:
                        combined_df['Genre'] = combined_df[col] # Assign the content to a new 'Genre' column
                        genre_column_found = col
                        break # Use the first valid potential genre column found

                if not genre_column_found:
                
                    combined_df['Genre'] = 'N/A' # Add a placeholder Genre column

                # Example: Handle potential duplicate rows after merge if necessary
                # combined_df = combined_df.drop_duplicates(subset=['Title', 'Author'])

                # Reset index after potential drops/merges
                combined_df = combined_df.reset_index(drop=True)

            except Exception as e:
                st.error(f"Error combining dataframes: {e}")
                st.warning("Returning loaded dataframes separately due to merge issue.")
                # If merging fails, try to concatenate with a source indicator
                try:
                    combined_df = pd.concat([df1.assign(source='file1'), df2.assign(source='file2')], ignore_index=True)
                    st.info("Data concatenated. Some visualizations may not be available if key columns are missing or from different sources.")
                # Add 'Genre' column based on available data if concatenation happens
                    if 'Genre_File1' in combined_df.columns: combined_df['Genre'] = combined_df['Genre_File1']
                    elif 'Tag_Column1' in combined_df.columns: combined_df['Genre'] = combined_df['Tag_Column1']
                    # ... check other potential genre columns
                    else: combined_df['Genre'] = 'N/A'


                except Exception as concat_e:
                    st.error(f"Error concatenating dataframes: {concat_e}")
                    combined_df = pd.DataFrame() # Ensure empty df if concat also fails


        else:
            st.warning("Cannot merge dataframes on 'Title' and 'Author' as required columns are missing, inconsistent, or empty. Concatenating data.")
            # If merge columns are missing or inconsistent, just concatenate with a source indicator
            try:
                combined_df = pd.concat([df1.assign(source='file1'), df2.assign(source='file2')], ignore_index=True)
                st.info("Data concatenated. Some visualizations may not be available if key columns are missing or from different sources.")
                # Add 'Genre' column based on available data if concatenation happens
                if 'Genre_File1' in combined_df.columns: combined_df['Genre'] = combined_df['Genre_File1']
                elif 'Tag_Column1' in combined_df.columns: combined_df['Genre'] = combined_df['Tag_Column1']
                # ... check other potential genre columns
                else: combined_df['Genre'] = 'N/A'

            except Exception as concat_e:
                st.error(f"Error concatenating dataframes: {concat_e}")
                combined_df = pd.DataFrame() # Ensure empty df if concat also fails


    else:
        st.error("No dataframes were loaded successfully to combine.")
        combined_df = pd.DataFrame() # Ensure empty DataFrame if both fail


    # --- Final Data Cleaning Steps on Combined DataFrame ---
    if not combined_df.empty:
        # Fill NaN in numeric rating/count columns with 0 if appropriate for analysis
        # This should happen AFTER consolidation
        for col in ['Avg Rating', 'Rating Count', 'Review Count']:
            if col in combined_df.columns and pd.api.types.is_numeric_dtype(combined_df[col]):
                combined_df[col] = combined_df[col].fillna(0)

        # Ensure essential display columns exist, fill with 'N/A' if missing after all steps
        essential_display_cols = ['Title', 'Author', 'Avg Rating', 'Rating Count', 'Review Count', 'Genre', 'Goodreads Link', 'Cover Image URL', 'Description']
        for col in essential_display_cols:
            if col not in combined_df.columns:
                combined_df[col] = 'N/A' # Add missing column with placeholder


    return combined_df

file1_csv_path = r"C:\Users\User\Onedrive\Desktop\streamlit\goodreads_csv.csv"
file2_csv_path = r"C:\Users\User\Onedrive\Desktop\streamlit\goodreads_data.csv"
# ------------------

# Load the data
combined_goodreads_df = load_goodreads_data(file1_csv_path, file2_csv_path)


st.set_page_config(page_title="BOOKY", layout="centered", page_icon="📚")

st.markdown(
    """
    <h1 style='text-align: center; font-family: "Cinzel", serif; color: ;'>
        BOOKY
    </h1>
    """,
    unsafe_allow_html=True
)
# Set the title of the app
 #Create tabs
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🇱🇧 Shop in Lebanon",
    "📚 Explore at Goodreads",
    "🔌 API",
    "❓Search by genre&price",
    "💬 Ask a Book a Question",
    "🎬 Book to Movie"
])

# ---------------------
# 🇱🇧 Shop in Lebanon
with tab1:
    st.subheader("🇱🇧 Shop in Lebanon")
    st.write("Compare book prices across Lebanese bookstores like Antoine, Koala Shop, and The Circle.")
    local_df = pd.read_csv(r"C:\Users\user\Onedrive\Desktop\streamlit\cleaned_books_data.csv")
    def parse_books(data):
        books = []
        for item in data:
            volume = item.get("volumeInfo", {})
            sale = item.get("saleInfo", {})

            # Try to extract price
            price = None
            if sale.get("saleability") == "FOR_SALE":
                price_info = sale.get("retailPrice", {})
                price = price_info.get("amount")

            books.append({
                "Title": volume.get("title", "N/A"),
                "Authors": ", ".join(volume.get("authors", [])),
                "Description": volume.get("description", "No description available."),
                "Rating": volume.get("averageRating", None),
                "Genre": ", ".join(volume.get("categories", [])) if volume.get("categories") else "N/A",
                "Published": volume.get("publishedDate", "N/A"),
                "Link": volume.get("infoLink", ""),
                "Price": price  # Add price here
            })
        return pd.DataFrame(books)



    query = st.text_input("Search by title or author:")
    if query:
        result = local_df[local_df['Title'].str.contains(query, case=False) | local_df['Author'].str.contains(query, case=False)]
        st.write(result)

    ### Analysis Visualizations

    # 📊 Price columns
    price_cols = {
        "Koala Shop": "Koala Shop Price",
        "Antoine Shop": "Antoine Shop Price",
        "The Circle Bookshop": "The Circle Bookshop Price"
    }


    # Custom style for header
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@600&display=swap');

        .custom-header {
            font-family: 'Cinzel', serif;
            font-size: 36px;
            text-align: center;
            color: black;
            padding: 12px 0;
        }
        </style>

        <div class="custom-header">📚 Bookstore Dashboard</div>
        """,
        unsafe_allow_html=True
    )


    # Visualization selector

    viz_options = [
        "Price Distribution Histogram",
        "Book Count per Shop",
        "Genre Distribution by Shop",
        "Book Title Word Cloud",
        "Genre Distribution Across All Shops"
    ]
    selected_viz = st.selectbox("Select a visualization", viz_options)

    # 👉 1. Price Distribution Histogram
    if selected_viz == "Price Distribution Histogram":
        selected_shop = st.selectbox("Select a bookstore:", list(price_cols.keys()))
        selected_col = price_cols[selected_shop]

        fig, ax = plt.subplots()
        sns.histplot(local_df[selected_col].dropna(), kde=True, ax=ax,
                    color='blue' if selected_shop == "Koala Shop" else
                        'green' if selected_shop == "Antoine Shop" else 'purple')
        ax.set_title(f'Price Distribution ({selected_shop})')
        ax.set_xlabel('Price')
        ax.set_ylabel('Frequency')
        st.pyplot(fig)

    # 👉 2. Book Count per Shop
    elif selected_viz == "Book Count per Shop":
        st.subheader("📦 Number of Books per Shop")
        book_counts = {shop: local_df[col].notna().sum() for shop, col in price_cols.items()}
        book_counts_df = pd.DataFrame(list(book_counts.items()), columns=["Shop", "Number of Books"])
        fig2, ax2 = plt.subplots()
        sns.barplot(data=book_counts_df, x="Shop", y="Number of Books", palette='Set2', ax=ax2)
        ax2.set_title("Number of Books Available per Bookstore")
        ax2.set_ylabel("Books Count")
        ax2.set_xlabel("Bookstore")
        st.pyplot(fig2)

    # 👉 3. Genre Distribution per Shop
    elif selected_viz == "Genre Distribution by Shop":
        st.subheader("🎭 Genre Distribution per Bookstore")
        selected_shop = st.selectbox("Choose bookstore:", list(price_cols.keys()))
        selected_col = price_cols[selected_shop]
        shop_books = local_df[local_df[selected_col].notna()]

        if 'Genre' in shop_books.columns:
            genre_counts = shop_books['Genre'].value_counts().reset_index()
            genre_counts.columns = ['Genre', 'Count']
            fig3, ax3 = plt.subplots(figsize=(8, 5))
            sns.barplot(data=genre_counts.head(10), x='Count', y='Genre', ax=ax3, palette='pastel')
            ax3.set_title(f"Top Genres in {selected_shop}")
            ax3.set_xlabel("Number of Books")
            ax3.set_ylabel("Genre")
            st.pyplot(fig3)
        else:
            st.info("Genre data not available in the dataset.")

    # 👉 4. Word Cloud of Book Titles
    elif selected_viz == "Book Title Word Cloud":
        st.subheader("📖 Word Cloud of Book Titles")
        text = " ".join(local_df['Title'].dropna())
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

    # 👉 5. Genre Distribution Across All Shops
    elif selected_viz == "Genre Distribution Across All Shops":
        st.subheader("📚 Genre Distribution (All Shops Combined)")
        if 'Genre' in local_df.columns:
            genre_counts = local_df['Genre'].value_counts()
            st.bar_chart(genre_counts)
        else:
            st.info("Genre column not found in the dataset.")

    st.markdown("---")


with tab2:
        
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Cinzel:wght@700&display=swap');

        .custom-title {
            font-family: 'Cinzel', serif;
            font-size: 25px;
            text-align: center;
            color: white;
            padding: 20px 0;
        }
        </style>

        <div class="custom-title">Explore your Book in goodreads!</div>
        """,
        unsafe_allow_html=True
    )

    if 'Title' in combined_goodreads_df.columns and combined_goodreads_df['Title'].dropna().empty == False:
        # Get unique non-NaN titles, sort them, and add a blank option at the beginning
        book_titles = combined_goodreads_df['Title'].dropna().unique()
        book_titles_sorted = sorted(book_titles)
        selected_title = st.selectbox("Select a book title:", [""] + book_titles_sorted) # Add blank option

        if selected_title: # Proceed only if a title is selected (not the blank option)
            # Use .copy() to avoid SettingWithCopyWarning
            book_details = combined_goodreads_df[combined_goodreads_df['Title'] == selected_title].iloc[0].copy() # Get the first row if duplicates exist

            st.subheader(f"Details for: {selected_title}")

            # Display details using columns and markdown
            detail_col1, detail_col2 = st.columns([1, 2])

            with detail_col1:
                # --- Image Display Logic (with 'Not Available' check) ---
                image_url = book_details.get('Cover Image URL') # Use .get to avoid key error
                # Check if the value is a non-empty string and doesn't look like the error placeholder
                if image_url and pd.notna(image_url) and isinstance(image_url, str) and image_url.lower() != 'not available' and (image_url.startswith('http') or image_url.startswith('https')):
                        # Use a fixed width for the image
                        st.image(image_url, caption=book_details.get('Title', 'Untitled'), width=150) # Use .get for Title
                else:
                    st.info("No cover image available.")
                # -------------------------------------------------------

            with detail_col2:
                st.markdown(f"**Author(s)**: {book_details.get('Author', 'N/A')}")
                if 'Avg Rating' in book_details and pd.notna(book_details['Avg Rating']):
                    st.markdown(f"**Average Rating**: {book_details['Avg Rating']:.2f} ⭐")
                if 'Rating Count' in book_details and pd.notna(book_details['Rating Count']):
                        st.markdown(f"**Rating Count**: {book_details['Rating Count']:,}")
                if 'Review Count' in book_details and pd.notna(book_details['Review Count']):
                        st.markdown(f"**Review Count**: {book_details['Review Count']:,}")
                if 'Genre' in book_details and pd.notna(book_details['Genre']): # Use the consolidated 'Genre'
                    st.markdown(f"**Genre**: {book_details['Genre']}")
                # Add other relevant details here, checking if columns exist and are not null
                if 'Goodreads Link' in book_details and book_details['Goodreads Link'] and pd.notna(book_details['Goodreads Link']):
                    st.markdown(f"[🔗 View on Goodreads]({book_details['Goodreads Link']})")

            if 'Description' in book_details and book_details['Description'] and pd.notna(book_details['Description']):
                st.markdown("**Description:**")
                st.write(book_details['Description']) # Use st.write for longer text
            else:
                    st.info("No description available.")

    else:
        st.info("'Title' column not found or is empty for exploring individual books.")
    ########## Visualization selector########
    viz_options = [
        "Distribution of Average Ratings", 
        "Popularity (Rating Count vs Avg Rating)", 
        "Genre Analysis", 
        "Top Rated Books (Highly Rated & Reviewed)"
    ]
    selected_viz = st.selectbox("Select a visualization", viz_options)

    # --- Visualization 1: Distribution of Average Ratings ---
    if selected_viz == "Distribution of Average Ratings":
        st.header("Distribution of Average Ratings")
        if 'Avg Rating' in combined_goodreads_df.columns and pd.api.types.is_numeric_dtype(combined_goodreads_df['Avg Rating']):
            fig, ax = plt.subplots(figsize=(10, 6))
            # Drop NaN values for plotting the distribution
            sns.histplot(combined_goodreads_df['Avg Rating'].dropna(), kde=True, ax=ax, bins=20)
            ax.set_title("Distribution of Average Goodreads Ratings")
            ax.set_xlabel("Average Rating")
            ax.set_ylabel("Number of Books")
            st.pyplot(fig)
        else:
            st.info("Distribution plot not available ('Avg Rating' column missing or not numeric).")

    # --- Visualization 2: Popularity (Rating Count vs Avg Rating) ---
    elif selected_viz == "Popularity (Rating Count vs Avg Rating)":
        st.header("Popularity: Rating Count vs Average Rating")
        if 'Avg Rating' in combined_goodreads_df.columns and pd.api.types.is_numeric_dtype(combined_goodreads_df['Avg Rating']) and \
        'Rating Count' in combined_goodreads_df.columns and pd.api.types.is_numeric_dtype(combined_goodreads_df['Rating Count']):

            # Use log scale for Rating Count, only for non-zero counts
            df_for_scatter = combined_goodreads_df[(combined_goodreads_df['Rating Count'] > 0) & (combined_goodreads_df['Rating Count'].notna())].copy()
            if not df_for_scatter.empty:
                # Ensure 'Rating Count' is numeric before applying log
                df_for_scatter['Rating Count (Log)'] = np.log10(df_for_scatter['Rating Count']) # Using log10 for easier interpretation

                fig, ax = plt.subplots(figsize=(10, 6))
                sns.scatterplot(data=df_for_scatter, x='Avg Rating', y='Rating Count (Log)', alpha=0.6, ax=ax)
                ax.set_title("Average Rating vs. Log10 of Rating Count (for books with >0 ratings)")
                ax.set_xlabel("Average Rating")
                ax.set_ylabel("Log10(Rating Count)")
                st.pyplot(fig)

                st.markdown("Books with more ratings tend to be clustered in certain rating ranges.")
            else:
                st.info("No books with ratings found for scatter plot.")
        else:
            st.info("Popularity scatter plot not available (check 'Avg Rating' and 'Rating Count' columns).")

    # --- Visualization 3: Genre Analysis ---
    elif selected_viz == "Genre Analysis":
        st.header("Genre Analysis")

        # --- Logic to identify the Genre/Tag column ---
        genre_column = None
        genre_column_options = ['Genre', 'Genre_File1', 'Tag_Column1', 'Tag_Column2', 'L', 'M']
        for col in genre_column_options:
            if col in combined_goodreads_df.columns and combined_goodreads_df[col].dropna().empty == False: # Check if column exists and is not empty
                genre_column = col
                st.markdown(f"Analysis based on column: **'{genre_column}'** (first non-empty potential genre column found).")
                break # Use the first valid genre column found

        if genre_column and genre_column in combined_goodreads_df.columns:
            # Drop rows with missing genre for counting and convert to string
            genre_data_for_counting = combined_goodreads_df[genre_column].astype(str).dropna()
            # Filter out empty strings and count
            genre_counts = genre_data_for_counting[genre_data_for_counting != ''].value_counts().head(20) # Top 20 genres

            if not genre_counts.empty:
                st.subheader("Top Genres/Tags by Book Count")
                fig, ax = plt.subplots(figsize=(12, 7))
                sns.barplot(x=genre_counts.values, y=genre_counts.index, ax=ax, palette='viridis')
                ax.set_title(f"Most Frequent {genre_column}")
                ax.set_xlabel("Number of Books")
                ax.set_ylabel(genre_column)
                st.pyplot(fig)

            else:
                st.info(f"No '{genre_column}' found with enough books for analysis.")
        else:
            st.info("No genre/tag information found for analysis after cleaning.")

    # --- Visualization 4: Top Rated Books (Highly Rated & Reviewed) ---
    elif selected_viz == "Top Rated Books (Highly Rated & Reviewed)":
        st.header("Top Rated Books (Highly Rated & Reviewed)")
        if 'Avg Rating' in combined_goodreads_df.columns and pd.api.types.is_numeric_dtype(combined_goodreads_df['Avg Rating']) and \
        'Rating Count' in combined_goodreads_df.columns and pd.api.types.is_numeric_dtype(combined_goodreads_df['Rating Count']):

            # Sidebar slider for minimum rating count threshold
            max_rating_count_present = combined_goodreads_df['Rating Count'].max() if combined_goodreads_df['Rating Count'].notna().any() else 0
            slider_max_value = int(max_rating_count_present) if max_rating_count_present > 0 else 10000 # Ensure a minimum if max is 0 or NaN
            slider_default_value = int(slider_max_value // 10) if slider_max_value > 10 else 0
            slider_step = int(slider_max_value // 100) if slider_max_value > 100 else 1
            if slider_step == 0 and slider_max_value > 0: slider_step = 1

            min_rating_count_for_top = st.sidebar.slider(
                "Minimum Rating Count for Top Books:",
                min_value=0, # Allow selecting from 0 ratings
                max_value=slider_max_value,
                value=slider_default_value,
                step=slider_step
            )

            # Filter books by minimum rating count and sort by average rating
            top_books = combined_goodreads_df[
                (combined_goodreads_df['Rating Count'] >= min_rating_count_for_top) & # Filter by count
                (combined_goodreads_df['Avg Rating'].notna()) # Ensure Average Rating is not NaN
            ].sort_values(by='Avg Rating', ascending=False).head(30) # Show Top 30

            if not top_books.empty:
                st.subheader(f"Top {len(top_books)} Books with at Least {min_rating_count_for_top:,} Ratings")
                display_cols = ['Title', 'Author', 'Avg Rating', 'Rating Count', 'Review Count', 'Genre', 'Goodreads Link']
                display_cols_present_and_useful = [col for col in display_cols if col in top_books.columns and top_books[col].notna().any()] 

                if display_cols_present_and_useful:
                    st.dataframe(top_books[display_cols_present_and_useful])
                else:
                    st.info("Key columns for displaying top books not found or are empty.")

                with st.expander("View Cover Images for Top Books"):
                    if 'Cover Image URL' in top_books.columns:
                        num_image_cols = 6
                        image_cols = st.columns(num_image_cols)
                        col_index = 0
                        for index, row in top_books.iterrows():
                            with image_cols[col_index]:
                                image_url = row['Cover Image URL']
                                if image_url and pd.notna(image_url) and isinstance(image_url, str) and image_url.lower() != 'not available' and (image_url.startswith('http') or image_url.startswith('https')):
                                    st.image(image_url, caption=row.get('Title', 'Untitled'), width=100)
                                else:
                                    st.write(f"No image for {row.get('Title', 'Book')[:20]}...")
                            col_index = (col_index + 1) % num_image_cols 

                    else:
                        st.info("'Cover Image URL' column not found for displaying covers.")
            else:
                st.info(f"No books found with at least {min_rating_count_for_top:,} ratings (after filtering).")
        else:
            st.info("Top Rated Books analysis not available (check 'Avg Rating' and 'Rating Count' columns).")


    # --- Overall Statistics ---
        st.subheader("Overall Statistics")
        col_stats1, col_stats2, col_stats3 = st.columns(3)
        col_stats1.metric("Total Books Loaded", len(combined_goodreads_df))

        # Calculate and display average rating if column exists and is numeric
        if 'Avg Rating' in combined_goodreads_df.columns and pd.api.types.is_numeric_dtype(combined_goodreads_df['Avg Rating']):
            avg_overall_rating = combined_goodreads_df['Avg Rating'].mean()
            col_stats2.metric("Average Rating", f"{avg_overall_rating:.2f} ⭐")

            # Calculate and display average rating count for rated books
            if 'Rating Count' in combined_goodreads_df.columns and pd.api.types.is_numeric_dtype(combined_goodreads_df['Rating Count']):
                # Filter out books with 0 or NaN ratings for meaningful average rating count
                filtered_df_for_counts = combined_goodreads_df[(combined_goodreads_df['Rating Count'] > 0) & (combined_goodreads_df['Rating Count'].notna())].copy()
                if not filtered_df_for_counts.empty:
                    avg_rating_count = filtered_df_for_counts['Rating Count'].mean()
                    col_stats3.metric("Average Rating Count (for rated books)", f"{avg_rating_count:,.0f}")
                else:
                    col_stats3.info("No books with >0 ratings for Avg Rating Count.")
            else:
                col_stats3.info("Rating Count column missing or not numeric for Avg Rating Count.")

        else:
            st.info("Rating statistics not available (check 'Avg Rating' column).")


# 🔌 API
with tab3:
    st.subheader("🔌 API")
 
    def search_books(query, max_results=20):
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("items", [])
        return []
    
    def parse_books(data):
        books = []
        for item in data:
            volume = item.get("volumeInfo", {})
            books.append({
                "Title": volume.get("title", "N/A"),
                "Authors": ", ".join(volume.get("authors", [])),
                "Description": volume.get("description", "No description available."),
                "Rating": volume.get("averageRating", None),
                "Published": volume.get("publishedDate", "N/A"),
                "Link": volume.get("infoLink", "")
            })
        return pd.DataFrame(books)
    
    query = st.text_input("Search for a book using Google Books / Open Library API:")
    if query:
        st.write(f"🔍 Searching APIs for: **{query}**...")
        results = search_books(query)
        
        if results:
            df_books = parse_books(results)
            st.dataframe(df_books[['Title', 'Authors', 'Rating']])
            
            # Show details for first result
            first_book = results[0]['volumeInfo']
            col1, col2 = st.columns([1, 2])
            
            with col1:
                if 'imageLinks' in first_book:
                    st.image(first_book['imageLinks']['thumbnail'], width=200)
            
            with col2:
                st.subheader(first_book.get('title', 'N/A'))
                st.write(f"**Authors**: {', '.join(first_book.get('authors', ['N/A']))}")
                st.write(f"**Rating**: {first_book.get('averageRating', 'N/A')}")
                st.write(f"**Published**: {first_book.get('publishedDate', 'N/A')}")
                if 'description' in first_book:
                    st.write(first_book['description'])
                if 'infoLink' in first_book:
                    st.markdown(f"[View on Google Books]({first_book['infoLink']})")
        else:
            st.warning("No results found.")

    local_df = pd.read_csv(r"C:\Users\user\Onedrive\Desktop\streamlit\cleaned_books_data.csv")
    def parse_books(data):
        books = []
        for item in data:
            volume = item.get("volumeInfo", {})
            sale = item.get("saleInfo", {})

            # Try to extract price
            price = None
            if sale.get("saleability") == "FOR_SALE":
                price_info = sale.get("retailPrice", {})
                price = price_info.get("amount")

            books.append({
                "Title": volume.get("title", "N/A"),
                "Authors": ", ".join(volume.get("authors", [])),
                "Description": volume.get("description", "No description available."),
                "Rating": volume.get("averageRating", None),
                "Genre": ", ".join(volume.get("categories", [])) if volume.get("categories") else "N/A",
                "Published": volume.get("publishedDate", "N/A"),
                "Link": volume.get("infoLink", ""),
                "Price": price  # Add price here
            })
        return pd.DataFrame(books)

    # --- Configurations
    GOOGLE_BOOKS_API_URL = "https://www.googleapis.com/books/v1/volumes?q="
    TMDB_API_KEY = "your_tmdb_api_key_here"
    TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
    TMDB_REVIEWS_URL = "https://api.themoviedb.org/3/movie/{movie_id}/reviews"


    # --- Book API Function (Google Books API)
    def search_books(query, max_results=20):
        url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults={max_results}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json().get("items", [])
        else:
            return []

        
    # --- Google Books Reviews
    def get_google_book_info(title):
        response = requests.get(f"{GOOGLE_BOOKS_API_URL}{title}")
        items = response.json().get("items", [])
        if not items:
            return None
        volume_info = items[0]["volumeInfo"]
        return {
            "title": volume_info.get("title", "N/A"),
            "rating": volume_info.get("averageRating", "N/A"),
            "description": volume_info.get("description", "No description available."),
        }
        
    # --- Book Price Comparison Function
    # --- Book Price Comparison Tool
    def compare_prices(book_title):
        local_books = local_df[local_df['Title'].str.contains(book_title, case=False)]
        google_books = search_books(book_title)
        
        if not local_books.empty and google_books:
            api_data = google_books[0]["volumeInfo"]
            book_info = {
                "Title": api_data.get("title", "N/A"),
                "Authors": ", ".join(api_data.get("authors", [])),
                "Google Price": api_data.get("price", {}).get("amount", "N/A"),
                "Rating": api_data.get("averageRating", "N/A"),
                "Google Link": api_data.get("infoLink", ""),
            }
            
            # Display local prices and highlight the cheapest option
            st.write("📘 Local Data")
            st.dataframe(local_books[['Title', 'Koala Shop Price', 'Antoine Shop Price', 'The Circle Bookshop Price']])
            
            min_price_row = local_books.loc[local_books[['Koala Shop Price', 'Antoine Shop Price', 'The Circle Bookshop Price']].min(axis=1).idxmin()]
            st.write(f"🛍 Cheapest Option: **{min_price_row['Koala Shop Price'] if min_price_row['Koala Shop Price'] == min_price_row[['Koala Shop Price', 'Antoine Shop Price', 'The Circle Bookshop Price']].min() else min_price_row['Antoine Shop Price'] if min_price_row['Antoine Shop Price'] == min_price_row[['Koala Shop Price', 'Antoine Shop Price', 'The Circle Bookshop Price']].min() else min_price_row['The Circle Bookshop Price']}**")

            st.write(f"🌐 Google Data: {book_info}")
        else:
            st.warning("Book not found in local data or Google Books.")



with tab4:
    st.subheader("❓Search by genre&price")
    # --- Search Section (CAN WE ADD OTHER GENERS?)
    genre = st.selectbox("Choose a genre", ["Fiction", "Fantasy", "Romance", "Thriller", "Self-Help", "Science"])
    max_price = st.slider("Max Price (USD)", 5, 100, 30)
    search_button = st.button("Search Books")

    if search_button:
        query = f"subject:{genre}"
        results = search_books(query, max_results=20)
        df_books = parse_books(results)

        # Filter by price
        df_filtered = df_books[df_books["Price"].notna()]
        df_filtered = df_filtered[df_filtered["Price"] <= max_price]

        if not df_filtered.empty:
            st.subheader("📖 Book Results")
            for i, row in df_filtered.iterrows():
                st.markdown(f"**{row['Title']}** by {row['Authors']}")
                st.markdown(f"⭐ Rating: {row['Rating']} | 💲 Price: ${row['Price']} | 📅 Published: {row['Published']}")
                st.markdown(f"[More Info]({row['Link']})")
                st.markdown("---")

            # --- Visualizations
            st.subheader("📊 Price vs Rating")
            fig, ax = plt.subplots()
            sns.scatterplot(data=df_filtered, x="Price", y="Rating", hue="Title", ax=ax)
            plt.xlabel("Price")
            plt.ylabel("Rating")
            st.pyplot(fig)

            st.subheader("📊 Books by Rating")
            fig2, ax2 = plt.subplots()
            sns.histplot(df_filtered["Rating"].dropna(), kde=True, bins=10, ax=ax2)
            plt.xlabel("Rating")
            plt.ylabel("Count")
            st.pyplot(fig2)
        else:
            st.warning("No books found within that price range.")

with tab5:
    st.subheader("💬 Ask a Book a Question")
    user_question = st.text_input("Ask (e.g., Best self-help books under $15):")

    if user_question:
        # Use the user question as a search query
        question_results = search_books(user_question, max_results=5)
        df_q = parse_books(question_results)
        if not df_q.empty:
            st.subheader("🔍 Results for your question:")
            for i, row in df_q.iterrows():
                st.markdown(f"**{row['Title']}** by {row['Authors']}")
                st.markdown(f"⭐ Rating: {row['Rating']} | 💲 Price: {row['Price']}")
                st.markdown(f"[More Info]({row['Link']})")
                st.markdown("---")
        else:
            st.info("No results found.")
    # Set your OpenAI API key
    openai.api_key = "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"

    # Google Books API function
    def get_google_books_data(title):
        url = f"https://www.googleapis.com/books/v1/volumes?q={title}"
        response = requests.get(url)
        if response.status_code == 200:
            items = response.json().get('items')
            if items:
                return items[0]['volumeInfo']
        return None
    # Open Library API function
    def get_open_library_data(isbn):
        url = f"https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&format=json&jscmd=data"
        response = requests.get(url)
        data = response.json()(f"ISBN:{isbn}", {}) if data else None
        return data

with tab6:
    def get_movie_adaptation(book_title):
        """
        Searches OMDb for a movie adaptation based on the book title.
        Returns movie data if found, otherwise None.
        """
        # --- YOUR ACTUAL OMDb API KEY ---
        # Key obtained from: https://www.omdbapi.com/apikey.aspx
        api_key = "c9f1fe13" # Your key has been inserted here
        # ---------------------------------

        if not api_key or api_key == "your_omdb_api_key":
            st.error("OMDb API key is missing or incorrect. Please get your key from https://www.omdbapi.com/apikey.aspx and insert it into the script.")
            return None

        # Use the 't' parameter to search by title
        url = f"http://www.omdbapi.com/?t={book_title}&apikey={api_key}"

        try:
            response = requests.get(url)
            response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            data = response.json()

            # Check if the API returned a result and if it's a movie
            if data.get("Response") == "True" and data.get("Type") == "movie":
                # Return relevant movie information
                return {
                    "Title": data.get("Title"),
                    "Year": data.get("Year"),
                    "Rated": data.get("Rated"),
                    "Genre": data.get("Genre"),
                    "Director": data.get("Director"),
                    "Actors": data.get("Actors"),
                    "Plot": data.get("Plot"),
                    "Poster": data.get("Poster"),
                    "imdbRating": data.get("imdbRating"),
                    "imdbID": data.get("imdbID"),
                    "Link": f"https://www.imdb.com/title/{data.get('imdbID')}" if data.get('imdbID') else None
                }
            elif data.get("Response") == "True" and data.get("Type") != "movie":
                # If a result was found but it's not a movie (e.g., a series, game)
                st.info(f"Found a result with a similar title, but it's a {data.get('Type')}, not a movie.")
                return None
            else:
                # Response is "False" - no result found with that title
                return None

        except requests.exceptions.RequestException as e:
            # Handle network errors, like no internet connection or invalid API key (sometimes shows as 401/403)
            st.error(f"Network error or API issue. Please check your internet connection and ensure your OMDb API key is active.")
            # You could print the error details for debugging: print(e)
            return None
        except json.JSONDecodeError:
            # Handle cases where the response is not valid JSON
            st.error("Error processing the API response from OMDb.")
            return None
        except Exception as e:
            # Catch any other unexpected errors
            st.error(f"An unexpected error occurred: {e}")
            return None

    st.markdown("Enter a book title in the box below to see if a movie adaptation is found in the OMDb database.")

    # Input field for the book title
    # The result will update automatically when the user types and pauses
    book_title_query = st.text_input("Book Title:", "").strip() # .strip() removes leading/trailing whitespace

    # Add a spinner while searching
    if book_title_query:
        # Only show spinner and search if input is not empty
        with st.spinner(f"Searching for a movie adaptation for '{book_title_query}'..."):
            movie_info = get_movie_adaptation(book_title_query) # Call the API function

        # --- Display Results ---
        if movie_info:
            st.subheader(f"✨ Movie Adaptation Found: {movie_info.get('Title', 'N/A')}")

            # Use columns for a n layout: Poster on the left, Details on the right
            col1, col2 = st.columns([1, 2]) # Adjust column ratios as needed

            with col1:
                # Display poster image if available and not "N/A"
                if movie_info.get("Poster") and movie_info.get("Poster") != "N/A":
                    st.image(movie_info["Poster"], caption=f"Poster for {movie_info.get('Title', 'Movie')}", use_container_width=True)
                else:
                    st.warning("No poster available for this movie.")

            with col2:
                st.markdown(f"**Year**: {movie_info.get('Year', 'N/A')}")
                st.markdown(f"**Genre**: {movie_info.get('Genre', 'N/A')}")

                # Display IMDB Rating with a check for availability
                if movie_info.get('imdbRating') and movie_info['imdbRating'] != "N/A":
                    st.markdown(f"**IMDB Rating**: {movie_info['imdbRating']} ⭐")
                else:
                    st.markdown(f"**IMDB Rating**: N/A") # Show N/A if rating is not available

                st.markdown(f"**Director(s)**: {movie_info.get('Director', 'N/A')}")
                st.markdown(f"**Actors**: {movie_info.get('Actors', 'N/A')}")

                # Display Plot
                st.markdown(f"**Plot**: {movie_info.get('Plot', 'N/A')}")

                # Add a link to the movie on IMDb
                if movie_info.get('Link'):
                    st.markdown(f"[🔗 View on IMDb]({movie_info['Link']})")

        elif movie_info is None and book_title_query != "":
            # This condition is met if get_movie_adaptation returns None and the input wasn't empty
            st.info(f"No movie adaptation found for '{book_title_query}' in the OMDb database.")
            st.caption("Note: The search looks for movies with titles similar to the book title. Some adaptations may have different titles or may not be listed in OMDb.")
        # else: # No input yet, or input was cleared - do nothing
    ##################################################################################ds
    st.markdown("---")
        # Set Streamlit page configuration for a wide layout

    # Load the data
    combined_goodreads_df = load_goodreads_data(file1_csv_path, file2_csv_path)

st.markdown("By Rana & Hawraa")
st.markdown("---")
