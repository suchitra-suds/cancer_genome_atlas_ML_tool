
import streamlit as st
import pandas as pd
import os
import subprocess
import time
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


# Run R script to get the list of projects

@st.cache_data
def get_projects():
    try:
        #  get project list from database
        result = subprocess.run(
            ["Rscript", "-e", "library(TCGAbiolinks); projects <- getGDCprojects(); projects <- projects[projects$releasable == \"TRUE\", ]; cat(paste(projects$id, collapse=','))"],
            capture_output=True,
            text=True
        )
        #  list of project IDs
        projects = result.stdout.strip().split(",")
        return projects
    except Exception as e:
        st.error(f"Error fetching projects: {e}")
        return []

# Function to get available data categories based on selected project
@st.cache_data
def get_data_categories(selected_project):
    try:
        #  get available data categories for the selected project
        result = subprocess.run(
            ["Rscript", "-e", f"library(TCGAbiolinks); available_queries <- getSampleFilesSummary(project = '{selected_project}'); available_queries <- available_queries[, !colnames(available_queries) %in% c('.id', 'project')]; column_names <- colnames(available_queries); split_columns <- strsplit(column_names, '_'); max_length <- max(sapply(split_columns, length)); split_columns_na <- lapply(split_columns, function(x) {{ length(x) <- max_length; return(x) }}); split_columns_df <- do.call(rbind, split_columns_na); colnames(split_columns_df) <- c('data_category', 'data_type', 'experimental_strategy', 'platform'); available_queries <- as.data.frame(split_columns_df); data_categories <- unique(available_queries$data_category); cat(paste(data_categories, collapse=','))"],
            capture_output=True,
            text=True
        )
        #  list of data categories
        data_categories = result.stdout.strip().split(",")
        return data_categories
    except Exception as e:
        st.error(f"Error fetching data categories: {e}")
        return []

#  get available data types based on selected data category
@st.cache_data
def get_data_types(selected_project, selected_data_category):
    try:
        #  get available data types for the selected data category
        result = subprocess.run(
            ["Rscript", "-e", f"library(TCGAbiolinks); available_queries <- getSampleFilesSummary(project = '{selected_project}'); available_queries <- available_queries[, !colnames(available_queries) %in% c('.id', 'project')]; column_names <- colnames(available_queries); split_columns <- strsplit(column_names, '_'); max_length <- max(sapply(split_columns, length)); split_columns_na <- lapply(split_columns, function(x) {{ length(x) <- max_length; return(x) }}); split_columns_df <- do.call(rbind, split_columns_na); colnames(split_columns_df) <- c('data_category', 'data_type', 'experimental_strategy', 'platform'); available_queries <- as.data.frame(split_columns_df); available_queries <- subset(available_queries, data_category == '{selected_data_category}'); data_types <- unique(available_queries$data_type); cat(paste(data_types, collapse=','))"],
            capture_output=True,
            text=True
        )
        # Split the output into a list of data types
        data_types = result.stdout.strip().split(",")
        return data_types
    except Exception as e:
        st.error(f"Error fetching data types: {e}")
        return []

#  get available experimental strategies based on selected data type
@st.cache_data
def get_experimental_strategies(selected_project, selected_data_category, selected_data_type):
    try:
        #  get available experimental strategies for the selected data type
        result = subprocess.run(
            ["Rscript", "-e", f"library(TCGAbiolinks); available_queries <- getSampleFilesSummary(project = '{selected_project}'); available_queries <- available_queries[, !colnames(available_queries) %in% c('.id', 'project')]; column_names <- colnames(available_queries); split_columns <- strsplit(column_names, '_'); max_length <- max(sapply(split_columns, length)); split_columns_na <- lapply(split_columns, function(x) {{ length(x) <- max_length; return(x) }}); split_columns_df <- do.call(rbind, split_columns_na); colnames(split_columns_df) <- c('data_category', 'data_type', 'experimental_strategy', 'platform'); available_queries <- as.data.frame(split_columns_df); available_queries <- subset(available_queries, data_category == '{selected_data_category}' & data_type == '{selected_data_type}'); experimental_strategies <- unique(available_queries$experimental_strategy); cat(paste(experimental_strategies, collapse=','))"],
            capture_output=True,
            text=True
        )
        #  list of experimental strategies
        experimental_strategies = result.stdout.strip().split(",")
        return experimental_strategies
    except Exception as e:
        st.error(f"Error fetching experimental strategies: {e}")
        return []

# get available platforms based on selected experimental strategy
@st.cache_data
def get_platforms(selected_project, selected_data_category, selected_data_type, selected_experimental_strategy):
    try:
        #  get available platforms for the selected experimental strategy
        result = subprocess.run(
            ["Rscript", "-e", f"library(TCGAbiolinks); available_queries <- getSampleFilesSummary(project = '{selected_project}'); available_queries <- available_queries[, !colnames(available_queries) %in% c('.id', 'project')]; column_names <- colnames(available_queries); split_columns <- strsplit(column_names, '_'); max_length <- max(sapply(split_columns, length)); split_columns_na <- lapply(split_columns, function(x) {{ length(x) <- max_length; return(x) }}); split_columns_df <- do.call(rbind, split_columns_na); colnames(split_columns_df) <- c('data_category', 'data_type', 'experimental_strategy', 'platform'); available_queries <- as.data.frame(split_columns_df); available_queries <- subset(available_queries, data_category == '{selected_data_category}' & data_type == '{selected_data_type}' & experimental_strategy == '{selected_experimental_strategy}'); platforms <- unique(available_queries$platform); cat(paste(platforms, collapse=','))"],
            capture_output=True,
            text=True
        )
        # Split the output into a list of platforms
        platforms = result.stdout.strip().split(",")
        return platforms
    except Exception as e:
        st.error(f"Error fetching platforms: {e}")
        return []

#  get sample types based on selected platform
@st.cache_data
def get_sample_types(selected_project, selected_data_category, selected_data_type, selected_experimental_strategy, selected_platform):
    try:
        # Run R script to get sample types for the selected platform
        result = subprocess.run(
            ["Rscript", "-e", f"library(TCGAbiolinks); query <- GDCquery(project = '{selected_project}', data.category = '{selected_data_category}', experimental.strategy = '{selected_experimental_strategy}', data.type = '{selected_data_type}', access = 'open'); query_results <- getResults(query); sample_types <- unique(query_results$sample_type); cat(paste(sample_types, collapse=','))"],
            capture_output=True,
            text=True
        )
        # Split the output into a list of sample types
        sample_types = result.stdout.strip().split(",")
        return sample_types
    except Exception as e:
        st.error(f"Error fetching sample types: {e}")
        return []

@st.cache_data
def download_data(selected_project, selected_data_category, selected_experimental_strategy, selected_data_type, selected_sample_types, number_of_cases, data_dir):
    try:

        progress_bar = st.progress(20)
        status_text = st.empty()

        # Run R script to perform the data query, download, and prepare gene metadata
        result = subprocess.run(
            [
                "Rscript", "-e", f"""
                library(TCGAbiolinks);
                library(SummarizedExperiment);
                library(DESeq2);
                query <- GDCquery(project = '{selected_project}', data.category = '{selected_data_category}', experimental.strategy = '{selected_experimental_strategy}', data.type = '{selected_data_type}', sample.type = c('{','.join(selected_sample_types)}'), access = 'open');
                query_results <- getResults(query);
                query_cases <- query_results[query_results$sample_type %in% c('{','.join(selected_sample_types)}'), 'cases'][1:{number_of_cases}];
                full_query <- GDCquery(project = '{selected_project}', data.category = '{selected_data_category}', experimental.strategy = '{selected_experimental_strategy}', data.type = '{selected_data_type}', sample.type = c('{','.join(selected_sample_types)}'), access = 'open', barcode = query_cases);
                GDCdownload(full_query, method = 'api', directory = '{data_dir}/GDCdata/', files.per.chunk = 50);
                data <- GDCprepare(full_query, summarizedExperiment = TRUE);
                matrix <- assay(data, 'unstranded');
                gene_metadata <- as.data.frame(rowData(data));
                coldata <- as.data.frame(colData(data));
                dds <- DESeqDataSetFromMatrix(countData = matrix, colData = coldata, design = ~ 1);
                keep <- rowSums(counts(dds)) >= 10;
                dds <- dds[keep,];
                vsd <- vst(dds, blind=FALSE);
                matrix_vst <- assay(vsd);
                write.csv(matrix_vst, file = file.path('{data_dir}', 'data_matrix.csv'))
                write.csv(gene_metadata, file = file.path('{data_dir}', 'gene_metadata.csv'))
                """
            ],
            capture_output=True,
            text=True
        )

        # Update progress bar to indicate completion
        progress_bar.progress(100)
        status_text.text("Processing complete.")

        if result.returncode == 0:
            st.success("Data download and processing complete.")
            output = result.stdout.strip().split('\n')
            query_cases = output[0].split(',')
            gene_list = output[1].split(',')
            print("Query Cases:", query_cases)

            st.session_state["query_cases"] = query_cases
            st.session_state["gene_list"] = gene_list

            # Write query cases to a text file
            query_cases_file = os.path.join(data_dir, "query_cases.txt")
            try:
                with open(query_cases_file, "w") as f:
                    f.write("\n".join(query_cases))
                st.success(f"Query cases saved to {query_cases_file}")
            except Exception as e:
                st.error(f"Failed to write query cases to file: {e}")
        else:
            st.error(f"Error during data download: {result.stderr}")
    except Exception as e:
        st.error(f"Error executing data download: {e}")


import streamlit as st
import subprocess
import os


st.set_page_config(layout="wide")
st.title("The Cancer Genome Atlas: Integrated Data Analysis and Machine Learning Platform")
tab1, tab2, tab3 = st.tabs(["⬇️ Data Download", "📊 Data Preview", "🔍 Analysis"])

with tab1:
    st.header("TCGA: Data Downloader")

    # Initialize session state flags for steps
    if 'steps' not in st.session_state:
        st.session_state['steps'] = {
            "number_of_cases_entered": False,
            "data_dir_set": False,
            "project_selected": False,
            "data_category_selected": False,
            "data_type_selected": False,
            "experimental_strategy_selected": False,
            "platform_selected": False,
            "sample_types_selected": False,
        }

    # Step 1: Enter Number of Cases
    st.write("### Step 1: Enter the Number of Cases")
    number_of_cases = st.number_input(
        "Number of cases:", min_value=1, step=1, key="number_of_cases"
    )
    if st.session_state.get("number_of_cases"):
        st.session_state['steps']['number_of_cases_entered'] = True
        st.write(f"**Selected number of cases:** {st.session_state['number_of_cases']}")

    # Step 2: Enter Data Directory
    if st.session_state['steps']['number_of_cases_entered']:
        st.write("### Step 2: Enter the Data Directory")
        data_dir = st.text_input("Directory path:", key="data_dir")
        if st.session_state.get("data_dir"):
            st.session_state['steps']['data_dir_set'] = True
            st.write(f"**Selected directory path:** {st.session_state['data_dir']}")

    # Step 3: Select Project
    if st.session_state['steps']['data_dir_set']:
        st.write("### Step 3: Select a TCGA Project")
        if "projects" not in st.session_state:
            st.session_state["projects"] = get_projects()

        selected_project = st.selectbox(
            "Select a project:", st.session_state["projects"], key="selected_project"
        )
        if selected_project:
            st.session_state['steps']['project_selected'] = True
            st.write(f"**Selected project:** {selected_project}")

    # Step 4: Select Data Category
    if st.session_state['steps']['project_selected']:
        st.write("### Step 4: Select a Data Category")
        if "data_categories" not in st.session_state or st.session_state["steps"].get("project_selected", False):
            st.session_state["data_categories"] = get_data_categories(st.session_state["selected_project"])

        selected_data_category = st.selectbox(
            "Select a data category:",
            st.session_state["data_categories"],
            key="selected_data_category",
        )
        if selected_data_category:
            st.session_state['steps']['data_category_selected'] = True
            st.write(f"**Selected data category:** {selected_data_category}")

    # Step 5: Select Data Type
    if st.session_state['steps']['data_category_selected']:
        st.write("### Step 5: Select a Data Type")
        if "data_types" not in st.session_state or st.session_state["steps"].get("data_category_selected", False):
            st.session_state["data_types"] = get_data_types(
                st.session_state["selected_project"],
                st.session_state["selected_data_category"],
            )

        selected_data_type = st.selectbox(
            "Select a data type:",
            st.session_state["data_types"],
            key="selected_data_type",
        )
        if selected_data_type:
            st.session_state['steps']['data_type_selected'] = True
            st.write(f"**Selected data type:** {selected_data_type}")

    # Step 6: Select Experimental Strategy
    if st.session_state['steps']['data_type_selected']:
        st.write("### Step 6: Select an Experimental Strategy")
        if "experimental_strategies" not in st.session_state or st.session_state["steps"].get("data_type_selected", False):
            st.session_state["experimental_strategies"] = get_experimental_strategies(
                st.session_state["selected_project"],
                st.session_state["selected_data_category"],
                st.session_state["selected_data_type"],
            )

        selected_experimental_strategy = st.selectbox(
            "Select an experimental strategy:",
            st.session_state["experimental_strategies"],
            key="selected_experimental_strategy",
        )
        if selected_experimental_strategy:
            st.session_state['steps']['experimental_strategy_selected'] = True
            st.write(f"**Selected experimental strategy:** {selected_experimental_strategy}")

    # Step 7: Select Platform
    if st.session_state['steps']['experimental_strategy_selected']:
        st.write("### Step 7: Select a Platform")
        if "platforms" not in st.session_state or st.session_state["steps"].get("experimental_strategy_selected", False):
            st.session_state["platforms"] = get_platforms(
                st.session_state["selected_project"],
                st.session_state["selected_data_category"],
                st.session_state["selected_data_type"],
                st.session_state["selected_experimental_strategy"],
            )

        selected_platform = st.selectbox(
            "Select a platform:", st.session_state["platforms"], key="selected_platform"
        )
        if selected_platform:
            st.session_state['steps']['platform_selected'] = True
            st.write(f"**Selected platform:** {selected_platform}")

    # Step 8: Select Sample Types
    if st.session_state['steps']['platform_selected']:
        st.write("### Step 8: Select Sample Types")
        if "sample_types" not in st.session_state or st.session_state["steps"].get("platform_selected", False):
            st.session_state["sample_types"] = get_sample_types(
                st.session_state["selected_project"],
                st.session_state["selected_data_category"],
                st.session_state["selected_data_type"],
                st.session_state["selected_experimental_strategy"],
                st.session_state["selected_platform"],
            )

        selected_sample_types = st.multiselect(
            "Select sample types:", st.session_state["sample_types"], key="selected_sample_types"
        )
        if selected_sample_types:
            st.session_state['steps']['sample_types_selected'] = True
            st.write(f"**Selected sample types:** {', '.join(selected_sample_types)}")

    # Step 9: Download Data
    if all(st.session_state['steps'].values()):
        st.write("### Step 9: Download Data")
        if st.button("Download Data"):
            try:
                download_data(
                    st.session_state["selected_project"],
                    st.session_state["selected_data_category"],
                    st.session_state["selected_experimental_strategy"],
                    st.session_state["selected_data_type"],
                    st.session_state["selected_sample_types"],
                    st.session_state["number_of_cases"],
                    st.session_state["data_dir"],
                )
                st.success("Data downloaded and processed successfully!")
            except Exception as e:
                st.error(f"Error during data download: {e}")




with tab2:
        # Display header for survival analysis section
        st.header("Data Selection")

        # Initialize session state flags for steps
        if 'steps' not in st.session_state:
            st.session_state['steps'] = {
                "gene_metadata_loaded": False,
                "matrix_info_loaded": False,
                "clinical_info_loaded": False,
                "data_merged": False
            }

        st.session_state['data_dir2'] = st.text_input("Enter the directory path to download and save files:")
        st.write(st.session_state['data_dir2'])

        # Define project parameters
        st.session_state['selected_project2'] = st.text_input("Enter the TCGA project name:")
        st.write(st.session_state['selected_project2'])
        progress_message = st.empty()

        # Step 2: Load gene metadata
        if not st.session_state['steps'].get('gene_metadata_loaded', False):
            if 'gene_metadata_path' not in st.session_state:
                progress_message.text('Gene metadata file not found. Triggering data download...')
                st.session_state['gene_metadata_path'] = os.path.join(st.session_state['data_dir2'], 'gene_metadata.csv')

            if 'gene_metadata' not in st.session_state:
                progress_message.text('Reading gene metadata...')
                try:
                    st.session_state['gene_metadata'] = pd.read_csv(st.session_state['gene_metadata_path'])
                    st.session_state['steps']['gene_metadata_loaded'] = True
                    progress_message.text('Gene metadata loaded successfully.')
                except FileNotFoundError:
                    st.error("Gene metadata file not found. Ensure the file exists in the specified directory.")
                    st.stop()

        # Step 3: Create gene list and select a gene
        if st.session_state['steps']['gene_metadata_loaded']:
            if 'gene_list' not in st.session_state:
                progress_message.text('Creating gene list for selection...')
                gene_metadata = st.session_state['gene_metadata']
                st.session_state['gene_list'] = gene_metadata['gene_name'].tolist()
                progress_message.text('Gene list created.')

            selected_gene = st.selectbox("Select a gene for analysis:", st.session_state['gene_list'])

            if selected_gene != st.session_state.get('selected_gene', None):
                st.session_state['selected_gene'] = selected_gene
                st.session_state.pop('matrix_info_filtered', None)  # Reset filtered matrix info
                st.session_state.pop('merged_data', None)  # Reset merged data
                st.session_state.pop('merged_data_filtered', None)  # Reset filtered merged data
                st.session_state['steps']['data_merged'] = False  # Mark merge step as incomplete
                st.session_state['steps']['matrix_info_loaded'] = False  # Reset matrix info flag
                st.session_state['steps']['clinical_info_loaded'] = False  # Reset clinical info flag

            st.write(f"Currently Selected Gene: {st.session_state['selected_gene']}")

        # Proceed only after gene is selected

        if 'selected_gene' not in st.session_state or not st.session_state['selected_gene']:
            st.warning("Please select a gene to proceed.")
            st.stop()

        # Step 4: Load matrix info
        if not st.session_state['steps']['matrix_info_loaded']:
            with st.spinner('Getting matrix path...'):
                if 'gene_matrix_path' not in st.session_state:
                    st.session_state['gene_matrix_path'] = os.path.join(st.session_state['data_dir2'], 'data_matrix.csv')

            gene_matrix_path = st.session_state['gene_matrix_path']


            with st.spinner('Processing matrix info in R...'):
                result = subprocess.run(
                    [
                        "Rscript", "-e", f"""
                        library(tidyverse);
                        gene_matrix <- read.csv('{gene_matrix_path}', row.names = 1);
                        gene_metadata <- read.csv('{os.path.join(st.session_state['data_dir'], 'gene_metadata.csv')}');
                        matrix_info <- gene_matrix %>% 
                        as.data.frame() %>%
                        rownames_to_column(var = 'gene_id') %>%
                        gather(key = 'case_id', value = 'counts', -gene_id) %>%
                        left_join(gene_metadata, by = "gene_id");
                        write.csv(matrix_info, file = file.path('{st.session_state['data_dir2']}', 'matrix_info.csv'), row.names = FALSE);
                        """
                    ],
                    capture_output=True,
                    text=True
            )

    # Step 4.3: Load the processed matrix info
            with st.spinner('Loading processed matrix info...'):
                data_dir = st.session_state['data_dir2']
                st.session_state['matrix_info_path'] = os.path.join(data_dir, 'matrix_info.csv')
                try:
                    st.session_state['matrix_info'] = pd.read_csv(st.session_state['matrix_info_path'])
                    st.session_state['steps']['matrix_info_loaded'] = True
                    st.success('Matrix info loaded successfully.')
                except FileNotFoundError:
                    st.error("Matrix info file not found. Ensure the R script executed correctly.")
                    st.stop()

        # Step 5: Filter matrix info by selected gene
        if st.session_state['steps']['matrix_info_loaded']:
            matrix_info = st.session_state['matrix_info']
            st.session_state['matrix_info_filtered'] = matrix_info[matrix_info['gene_name'] == st.session_state['selected_gene']]

        # Step 6: Load clinical info
        if not st.session_state['steps']['clinical_info_loaded']:
            selected_project2=st.session_state['selected_project2']
            progress_message.text(f"Fetching clinical data...")
            result = subprocess.run(
                [
                    "Rscript", "-e", f"""
                    library(TCGAbiolinks);
                    clinical_info <- GDCquery_clinic('{selected_project2}');
                    write.csv(clinical_info, file.path('{st.session_state['data_dir2']}', 'clinical_info.csv'), row.names = FALSE);
                    """
                ],
                capture_output=True,
                text=True
            )
            clinical_info_path = os.path.join(st.session_state['data_dir2'], 'clinical_info.csv')
            try:
                st.session_state['clinical_info'] = pd.read_csv(clinical_info_path).dropna(axis=1, how='all')
                st.session_state['steps']['clinical_info_loaded'] = True
                progress_message.text('Clinical Info loaded successfully.')
            except FileNotFoundError:
                st.error("Clinical info file not found. Ensure the file exists in the specified directory.")
                st.stop()

        # Step 7: Merge matrix and clinical data
        if (
            st.session_state['steps']['matrix_info_loaded'] and 
            st.session_state['steps']['clinical_info_loaded'] and 
            not st.session_state['steps']['data_merged']
        ):
            progress_message.text('Merging matrix and clinical info...')
            matrix_info_filtered = st.session_state['matrix_info_filtered']
            matrix_info_filtered['case_id'] = (
                matrix_info_filtered['case_id']
                .str.replace(r'.01.*', '', regex=True)
                .str.replace(r'\.', '-', regex=True)
            )
            st.session_state['merged_data'] = pd.merge(
                matrix_info_filtered,
                st.session_state['clinical_info'],
                left_on='case_id',
                right_on='submitter_id',
                how='inner'
            )
            st.session_state['steps']['data_merged'] = True
            st.write("Merged Dataset Preview:")
            st.write(st.session_state['merged_data'].head())
            progress_message.text('Merged matrix and clinical info.')

        # Step 8: Select outcome and predictor variables
        if st.session_state['steps']['data_merged']:
            merged_data = st.session_state['merged_data']
            st.session_state['outcome'] = st.multiselect("Select outcome variables for analysis:", merged_data.columns.tolist())

            st.session_state['predictors'] = st.multiselect("Select predictor variables for analysis:", merged_data.columns.tolist())

            if st.session_state['outcome'] and st.session_state['predictors']:
                outcome = st.session_state['outcome']
                predictors = st.session_state['predictors']
                st.session_state['merged_data_filtered'] = merged_data[outcome + predictors]
                st.write('Current Data Set:')
                st.write(st.session_state['merged_data_filtered'].head())

        if st.session_state.get('merged_data_filtered') is not None:
                st.subheader("Visualized Data:")

    # Get the list of variables to plot
                selected_vars = st.session_state['outcome'] + st.session_state['predictors']
                num_vars = len(selected_vars)

    # Create a grid of columns based on the number of variables
                num_columns = 2  # Number of charts per row
                columns = st.columns(num_columns)
                sns.set_theme(style="whitegrid")

                for i, variable in enumerate(selected_vars):
                    if variable in st.session_state['merged_data_filtered'].columns:
                        col_idx = i % num_columns  # Determine which column to use
                        with columns[col_idx]:  # Place chart in the corresponding column
                            data = st.session_state['merged_data_filtered'][variable]

                            if data.dtype in ['float64', 'int64']:  # Continuous variable: Histogram
                                st.write(f"Histogram of {variable}")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.histplot(data, kde=True, ax=ax, color="teal")
                                if {variable} == 'counts':
                                    ax.set_title(f"Histogram of Variance Stabilized Tranformed Counts")
                                else:
                                    ax.set_title(f"Histogram of {variable}")
                                ax.set_xlabel(variable)
                                ax.set_ylabel("Frequency")
                                st.pyplot(fig)

                            else:  # Categorical variable: Bar chart
                                value_counts = data.value_counts()
                                st.write(f"Bar Chart of {variable}")
                                fig, ax = plt.subplots(figsize=(6, 4))
                                sns.barplot(
                                    x=value_counts.values,
                                    y=value_counts.index,
                                    ax=ax,
                                    palette=sns.color_palette("viridis", len(value_counts)),
                                    edgecolor="black"
                                )
                                # Add data labels
                                for idx, value in enumerate(value_counts.values):
                                    ax.text(value + 0.5, idx, str(value), va="center", fontsize=10)

                                ax.set_title(f"Bar Chart of {variable}", fontsize=14, fontweight="bold")
                                ax.set_xlabel("Counts", fontsize=12)
                                ax.set_ylabel("Values", fontsize=12)
                                ax.grid(color="gray", linestyle="--", linewidth=0.5, alpha=0.7)
                                st.pyplot(fig)


                # Encode categorical variables
                st.subheader("Encoded Variables")
                encoded_data = st.session_state['merged_data_filtered'].copy()
                encoding_legend = {}

                for column in encoded_data.columns:
                    if encoded_data[column].dtype == 'object':
                        unique_values = encoded_data[column].unique()
                        mapping = {value: idx for idx, value in enumerate(unique_values)}
                        encoding_legend[column] = mapping
                        encoded_data[column] = encoded_data[column].map(mapping)

        # Display the encoded dataset and legend
                st.write("Encoded Dataset:")
                st.write(encoded_data.head())

                st.write("Encoding Legend:")
                for column, mapping in encoding_legend.items():
                    st.write(f"{column}: {mapping}")
                
                st.session_state['encoded_data'] = encoded_data


with tab3:
    st.subheader("Machine Learning Analysis")

    if st.session_state.get('encoded_data') is not None:
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report, accuracy_score, ConfusionMatrixDisplay
        import seaborn as sns

        if len(st.session_state['outcome']) == 1:
            outcome = st.session_state['outcome'][0]
            predictors = st.session_state['predictors']
            X = st.session_state['encoded_data'][predictors]
            y = st.session_state['encoded_data'][outcome]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train model
            model = RandomForestClassifier(random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display Model Performance
            st.markdown("### Model Performance")
            accuracy = accuracy_score(y_test, y_pred)
            st.metric(label="Model Accuracy", value=f"{accuracy:.2%}")
            st.write("Classification Report:")
            st.text(classification_report(y_test, y_pred))

            # Confusion matrix and feature importance side-by-side
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Confusion Matrix")
                fig, ax = plt.subplots(figsize=(6, 4))
                ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, ax=ax, cmap="viridis", colorbar=False)
                st.pyplot(fig)

            with col2:
                st.markdown("#### Feature Importance")
                feature_importances = pd.DataFrame({
                    'Feature': predictors,
                    'Importance': model.feature_importances_
                }).sort_values(by='Importance', ascending=False)
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.barplot(data=feature_importances, x='Importance', y='Feature', palette="coolwarm", ax=ax)
                ax.set_title("Feature Importance", fontsize=16)
                ax.set_xlabel("Importance", fontsize=12)
                ax.set_ylabel("Feature", fontsize=12)
                st.pyplot(fig)

            st.markdown("---") 

            # Overall Interpretation
            with st.expander("🔍 Model Interpretation", expanded=True):
                st.markdown("### Overall Model Insights")
                classification_report_data = classification_report(y_test, y_pred, output_dict=True)

                accuracy = classification_report_data['accuracy']
                weighted_avg_precision = classification_report_data['weighted avg']['precision']
                weighted_avg_recall = classification_report_data['weighted avg']['recall']
                macro_avg_f1 = classification_report_data['macro avg']['f1-score']

                st.write(f"**Accuracy:** {accuracy:.2%}")
                if accuracy >= 0.8:
                    st.success("The model performs well with high accuracy.")
                elif 0.5 <= accuracy < 0.8:
                    st.warning("The model shows moderate accuracy. Improvements may be needed.")
                else:
                    st.error("The model struggles with low accuracy. Consider revisiting the features or data quality.")

                st.write(f"**Weighted Average Precision:** {weighted_avg_precision:.2%}")
                st.write(f"**Weighted Average Recall:** {weighted_avg_recall:.2%}")
                st.write(f"**Macro Average F1 Score:** {macro_avg_f1:.2%}")

            # Class-Specific Insights in Table Format
            with st.expander("📊 Class-Specific Insights"):
                st.markdown("### Class-Specific Performance Metrics")
                class_insights = []
                for label, metrics in classification_report_data.items():
                    if isinstance(metrics, dict):  # Skip overall metrics like 'accuracy'
                        class_insights.append({
                            "Class": label,
                            "Precision": metrics.get('precision', 0.0),
                            "Recall": metrics.get('recall', 0.0),
                            "F1 Score": metrics.get('f1-score', 0.0),
                            "Support": metrics.get('support', 0)
                        })

                class_insights_df = pd.DataFrame(class_insights)
                st.dataframe(class_insights_df.style.format({
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1 Score": "{:.2%}"
                }))

            st.markdown("---")  # Separator

            # Improvements
            with st.expander("💡 Suggestions for Improvement"):
                st.markdown("### Suggestions for Model Improvement")
                if accuracy < 0.7 or macro_avg_f1 < 0.7:
                    st.write("- **Balance the dataset** to handle class imbalances.")
                    st.write("- **Optimize hyperparameters** using techniques like grid search.")
                    st.write("- **Improve feature engineering** to capture relevant patterns.")
                    st.write("- **Increase the training data size** or improve its quality.")
                else:
                    st.write("The model performs well overall. Minor improvements may further enhance performance.")
        else:
            st.warning("Please select only one outcome variable for machine learning analysis.")
