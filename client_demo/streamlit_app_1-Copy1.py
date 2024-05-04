import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (confusion_matrix, precision_recall_curve, auc, roc_curve, recall_score, classification_report, f1_score,precision_recall_fscore_support, roc_auc_score,accuracy_score, precision_score, recall_score, f1_score)
import webbrowser
import os
from ydata_profiling import ProfileReport
# from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import sweetviz as sv
from bs4 import BeautifulSoup
import subprocess
def load_data(file):
    return pd.read_csv(file, encoding='utf-8')
def float_to_int(x):
    return int(x)
def pandas_profiling_tab():
    st.subheader("Pandas Profiling")
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = load_data(uploaded_file)
        # pd.set_option('display.float_format', lambda x: '%.3f' % x)
        # df = df[['BASE_CCY_AMT', 'FNCT_CCY_AMT', 'TXN_CCY_AMT', 'IB_FLAG', 'TXN_CCY', 'OPER_DPST_FLAG', 'LV_DMC_CTRY', 'FRS_ACCOUNT_CLASS','ACCT_TYP', 'ACTG_UNIT_NM', 'ROW_SRC_IND', 'REC_TYP', 'ARRG_MAT_TYP_CD']]
        # df = df[['BASE_CCY_AMT', 'FNCT_CCY_AMT', 'TXN_CCY_AMT']]
        # df = df[['BASE_CCY_AMT','FNCT_CCY_AMT','TXN_CCY_AMT','TXN_CCY','IB_FLAG', 'OPER_DPST_FLAG','LV_DMC_CTRY', 'FRS_ACCOUNT_CLASS','ACCT_TYP', 'ACTG_UNIT_NM', 'ROW_SRC_IND','REC_TYP', 'ARRG_MAT_TYP_CD']]
        #df = df[['BASE_CCY_AMT','TXN_CCY_AMT','TXN_CCY','IB_FLAG', 'OPER_DPST_FLAG','LV_DMC_CTRY', 'FRS_ACCOUNT_CLASS','ACCT_TYP', 'ACTG_UNIT_NM', 'ROW_SRC_IND','REC_TYP', 'ARRG_MAT_TYP_CD']]
        #df = df.map(lambda x: int(x))
        # df = df.applymap(float_to_int)
        # float_cols = df.select_dtypes(include=['float64']).columns
        # df[float_cols] = df[float_cols].astype('float32').round(2)
        profile = ProfileReport(df, explorative=True)
        st_profile_report(profile)
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, encoding_dim)
        self.decoder = nn.Linear(encoding_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# Calculate threshold for anomaly detection
def calculate_threshold(errors, percentile):
    return np.percentile(errors, percentile)

# Perform anomaly detection using the trained autoencoder
def detect_anomalies(model, test_data, threshold):
    test_data = torch.from_numpy(test_data.values).float()
    with torch.no_grad():
        reconstructed_data = model(test_data)
        mse_loss = torch.mean(torch.pow(test_data - reconstructed_data, 2), dim=1)
        anomalies = np.where(mse_loss > threshold)[0]
    return anomalies
def graph_img(metrics_html, html_table, X, df, shap_value_instances, explainer, feature_df,shap_values,css_content):
    
    css ='''  .centert {     display: flex;     justify-content: center;     align-items: center;        flex-direction: column; }'''
    st.markdown(
        f'''
        <style>
        {css}
        </style>
        ''',
        unsafe_allow_html = True
        )
    metrics_html = metrics_html.replace('<table', '<table class="centert"')

    st.markdown(metrics_html, unsafe_allow_html=True)
    
    # st.subheader("Feature Importance Analysis")
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(feature_df)), feature_df['Importance'], tick_label=feature_df['Feature'])
    plt.xlabel('Feature')
    plt.ylabel('Importance')
    # plt.title('Feature Importance Analysis')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('feature_importance_plot.png')
    plt.close()

    # Display feature importances plot
    # st.image('feature_importance_plot.png')
    # st.subheader("SHAP Feature Importance for Global Explanation")
    st.markdown('''
    <div style="display: flex; justify-content: center;">
        <h3><br><br><br>SHAP Feature Importance for Global Explanation</h3>
    </div>''', unsafe_allow_html=True)

    # Plot SHAP values
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type="bar", show=False)
    # plt.title('SHAP Feature Importance for Global Explanation', fontsize=16)
    plt.xlabel('SHAP Value', fontsize=14)
    plt.ylabel('Feature', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()
    handles, labels = plt.gca().get_legend_handles_labels()
    labels = ['Normal', 'Anomalous']

    # Update legend
    plt.legend(handles, labels)
    plt.savefig('shap_summary_plot.png')
    plt.close()

    # Display SHAP values plot
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image('shap_summary_plot.png', width=600)
    
    # st.subheader("Anomalous Data")
    st.markdown('''
    <div style="display: flex; justify-content: center;">
        <h3><br><br><br>Anomalous Data</h3>
    </div>''', unsafe_allow_html=True)
    
    st.markdown(
        f'''
        <style>
        {css_content}
        </style>
        ''',
        unsafe_allow_html = True
        )
    st.markdown(html_table, unsafe_allow_html=True)
def Anomaly_Detection():
    st.markdown('''
    <div style="display: flex; justify-content: center;">
        <h1>Anomaly Detection, using Deep Learning, XAI<br>(Cash & Due Data)</h1>
        <h1><br><br></h1>
    </div>''', unsafe_allow_html=True)

    # st.subheader("Anomaly Summary")
    st.markdown('''
    <div style="display: flex; justify-content: center;">
        <h3><br><br>Anomaly Summary</h3>
    </div>''', unsafe_allow_html=True)
    
    
    df = pd.read_csv('AI_sample1.csv')

    df=df[['BASE_CCY_AMT', 'FNCT_CCY_AMT', 'TXN_CCY_AMT', 'IB_FLAG', 'TXN_CCY', 'OPER_DPST_FLAG', 'LV_DMC_CTRY', 'FRS_ACCOUNT_CLASS','ACCT_TYP', 'ACTG_UNIT_NM', 'ROW_SRC_IND', 'REC_TYP', 'ARRG_MAT_TYP_CD']]
    pd.set_option('float_format', '{:f}'.format)

    float_data = df.select_dtypes(include=['float','int'])

    ## Convert Categorical variables to factors
    # It encodes Categorical variables into numerical factors using LabelEncoder and stores the encoders in a dictionary.
    label_encoders = {}
    for column in df.select_dtypes(include=['object']):
        encoder = LabelEncoder()
        df[column] = encoder.fit_transform(df[column])
        label_encoders[column] = encoder
    scaler = MinMaxScaler()
    df =df.drop([],axis=1)
    test_df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)

    # Load the pre-trained model
    model = Autoencoder(input_dim=test_df_scaled.shape[1], encoding_dim=10)
    model.load_state_dict(torch.load('autoencoder_model_1.pth'))
    model.eval()

    # Perform anomaly detection to calculate MSE for test dataset
    with torch.no_grad():
        reconstructed_data = model(torch.from_numpy(test_df_scaled.values).float())#d2
        mse_loss = torch.mean(torch.pow(torch.from_numpy(test_df_scaled.values).float() - reconstructed_data, 2), dim=1) #d1

    # Calculate threshold for anomaly detection
    threshold = calculate_threshold(mse_loss.numpy(), percentile=95)

    # Perform anomaly detection using the calculated threshold
    anomalies = detect_anomalies(model, test_df_scaled, threshold)
    new_df = df.copy()
    new_df['Anomaly_Found'] = new_df.index.isin(anomalies).astype(int)
    new_df.to_csv('pp.csv', index=False) 
    # per_df= test_data.copy()
    # per_df['Anomaly_Found'] =0
    # per_df.iloc[anomalies,-1]=1
    X = new_df.drop(columns=['Anomaly_Found'])
    y = new_df['Anomaly_Found']
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X, y)

    # Get feature importances
    feature_importances = clf.feature_importances_
    feature_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importances})
    feature_df = feature_df.sort_values(by='Importance', ascending=False)
    # Use SHAP to explain the model's predictions
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X)
    anomalies = new_df[new_df['Anomaly_Found'] == 1]
    anomalies_rows=len(anomalies)
    test_dataset_rows = len(df)
    per=round(((anomalies_rows/test_dataset_rows)*100),2)
    error_df = pd.DataFrame({'reconstruction_error': mse_loss, 'true_class':y})
    css_content = '''
    table {
        border-collapse: collapse;
        margin: 20px auto;
        overflow-y:scroll;
        display:block;
        border: none;
    }
    th, td {
        padding: 8px;
        text-align: center;
        border-bottom: 1px solid #ddd;
    }
    th{
        background-color: #007bff;
        color: #fff; /* White text color for heading */
    }
    .blue-row {
        background-color: #cceeff; /* Blue color */
    }
    .white-row {
        background-color: #ffffff; /* White color */
    }
    '''
    css = '''  .center {     display: flex;     justify-content: center;     align-items: center;     height: 100vh;     flex-direction: column; }'''
    css_content = css_content + css
    html_table = anomalies.to_html(index=True, header=True, classes='my-table')
    html_table = html_table.replace('<table', '<table style="width: 1200px; height: 600px; border-collapse: collapse;"')
    html_table = html_table.replace('<tbody>', '<tbody>')

    # Alternating row colors
    for i, row in enumerate(df.iterrows()):
        if i % 2 == 0:
            html_table = html_table.replace(f'<tr>', f'<tr class="blue-row">', 1)
        else:
            html_table = html_table.replace(f'<tr>', f'<tr class="white-row">', 1)

    # Calculate metrics
    threshold_copy = str(round(threshold, 2))
    with open("style.css","w") as css_file:
        css_file.write(css_content)
    explainer = shap.TreeExplainer(clf)
    shap_value_instances = explainer.shap_values(X) 
    # Assuming you have true labels for anomalies in your dataset
    true_labels = new_df['Anomaly_Found']
 
    # Assuming you have predictions from your model
    # You might need to adjust this depending on how you're obtaining predictions
    # predicted_labels = clf.predict(X)
    df2 = pd.read_csv("standard.csv")
    predicted_labels= df2['Outliers']
    # Calculate accuracy
    accuracy = accuracy_score(true_labels, predicted_labels)
 
    # Calculate precision
    precision = precision_score(true_labels, predicted_labels)
 
    # Calculate recall
    recall = recall_score(true_labels, predicted_labels)
 
    # Calculate F1-score
    f1 = f1_score(true_labels, predicted_labels)
    metrics_data = {
        # 'Metric': ['No of rows in the record:', 'No of anomalies rows in the record:', 'What percentage of data is anomalous:', 'Mean Square Error Threshold'],
        'Metric': ['No of rows in the record:', 'No of anomalies rows in the record:', 'What percentage of data is anomalous:','Accuracy','Precision','Recall','F1-score'],
        # 'Metric': ['No of rows in the record:', 'No of anomalies rows in the record:', 'What percentage of data is anomalous:'],
        # 'Value': [test_dataset_rows, anomalies_rows, f'{per}%', threshold_copy]
        'Value': [test_dataset_rows, anomalies_rows, f'{per}%',accuracy,precision,recall,f1]
        # 'Value': [test_dataset_rows, anomalies_rows, f'{per}%']
    }
    metrics_df = pd.DataFrame(metrics_data)
    metrics_html = metrics_df.to_html(index=False, header=False, classes='my-table')
    metrics_html = metrics_html.replace('<table', '<table class="table table-striped" id="my-table"')
    graph_img(metrics_html, html_table, X, df, shap_value_instances, explainer, feature_df,shap_values,css_content)
    two_d_list = [item for sublist in shap_value_instances for item in sublist]
    shap_value = pd.DataFrame(two_d_list)
    shap_abs = abs(shap_value)
    a = new_df.drop(columns=['Anomaly_Found'])
    result_df = pd.DataFrame(0, index=range(len(new_df)), columns=a.columns)
    for i in range(len(new_df)):
        shap_row = shap_abs.iloc[i]
        top3_features_indices = shap_row.argsort()[:-6:-1]  # Get indices of top 3 features
        result_df.loc[i, a.columns[top3_features_indices]] = 1
    result_df['index']=new_df.index
    result_df = result_df.iloc[anomalies.index]
    result_df.to_csv('anomaly_features.csv', index=False)    
    y_pred =predicted_labels

    # # Create a confusion matrix
    # cm = confusion_matrix(true_labels, y_pred)

    # # Plot the confusion matrix
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    # plt.xlabel("Predicted labels")
    # plt.ylabel("True labels")
    # plt.title("Confusion Matrix")
    # plt.savefig('Confusion_matrix.png')
    # plt.clf()
    # st.image('Confusion_matrix.png')
    LABELS = ['Normal', 'Fraud']
    # y_pred = [1 if e > threshold else 0 for e in error_df.reconstruction_error.values]
    conf_matrix = confusion_matrix(true_labels, y_pred)
    sns.heatmap(conf_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d", cmap='Blues');
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.savefig('Confusion_matrix.png')
    plt.clf()
    st.image('Confusion_matrix.png')
    
    st.title('Search Row Number')
    instance_index = st.number_input('Enter Row Number:', min_value=0, max_value=len(df)-1, value=0, step=1)
    def plot_local_shap_explanation(instance_index):
        instances = X.iloc[[instance_index]]
        # Plot the waterfall plot
        shap.waterfall_plot(shap.Explanation(values=shap_value_instances[1][instance_index], base_values=explainer.expected_value[1], data=instances.iloc[0]), max_display=10)
        plt.title('SHAP Waterfall Plot')  # Add a title
        plt.tight_layout()
        plt.savefig('shap_local_explanation.png')
        plt.clf()
    if st.button('Search'):
        plot_local_shap_explanation(instance_index)
        st.title(f'SHAP Local Explanation for Row {instance_index}')
        left_co, cent_co,last_co = st.columns(3)
        with cent_co:
            st.image('shap_local_explanation.png',width = 700)
if __name__ == '__main__':
    command = ['pip', 'install', 'shap==0.42.1']
    try:
        subprocess.check_call(command)
        st.set_page_config(layout='wide')
        # Load your preprocessed test dataset
        #tabs =["Anomaly Detection","Pandas Profiling Report"]
        st.title("Data Quality Platform")
        tab1, tab2 = st.tabs(["Pandas Profiling Report","Anomaly Detection"])#"Sweetviz Report"])
        # User Input
        with tab1:
            pandas_profiling_tab()
        with tab2:
            Anomaly_Detection()
        
        # with tab3:
        #     df = load_data('pp.csv')
        #     report = sv.analyze(df."Anomaly_Found")
        #     # Display Sweetviz report in Streamlit
        #     st.write(report.show_html(), unsafe_allow_html=True)
    except subprocess.CalledProcessError:
        st.title("404 not found!")
        
        
