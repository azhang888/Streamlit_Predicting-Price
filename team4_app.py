#from turtle import color
import streamlit as st
import assignment1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from google.cloud import firestore

# connect to firestore
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = 'macro-key-340706-658d688f339a.json'
db = firestore.Client()

#################################################################
################## Page Settings ################################
#################################################################

st.set_page_config(page_title="Team 4", layout="wide", page_icon="📱",)
st.markdown('''
<style>
    #MainMenu
    {
        display: none;
    }
    .css-18e3th9, .css-1d391kg
    {
        padding: 1rem 2rem 2rem 2rem;
    }
</style>
''', unsafe_allow_html=True)


#################################################################
################## Page Header ##################################
#################################################################
st.title('Predicting Price Range of Mobile by Features')
st.caption("Team4: Siyu Du, Yiwei Liu, Aoran Zhang, Yifan Li")
st.markdown('---')
st.sidebar.image(("phone.jpg"), use_column_width=True)

################## Sidebar Menu #################################
page_selected = st.sidebar.radio("Menu", ["Home", "Model", "About"])

X, y = assignment1.load_prepare()
training_accuracy, confusion_matrix, pipeline = assignment1.build_pipeline_final(X, y)
# fetch data from firestore
docs = db.collection(u'predict').document(u'my_doc').get()
items = docs.to_dict()
df = pd.DataFrame.from_records(items)

################################################################
################## Home Page ###################################
################################################################
if page_selected == 'Home':    
    st.subheader('Backgroud')
    st.write('Bob has started his own mobile company. He wants to give tough fight to big companies like Apple, Samsung etc. He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem we collects training and test dataset of sales data of mobile phones of various companies from kaggles(Kaggle, 2022).')
    st.write('Our team tried to find out some relations among all these mobiel phone features. Besides our main story plots, we also explore the relationship between variables the audience might be interested in; Moreover, to pursue a better data performance, we used muitple types of charts. Our work is divided into 2 main parts: data visualization and predict modeling  ')
    st.write(':smile:', 'Through data cleaning, pipeline creating, model training, etc, I predict for the test data. The application is to show some analysis of prediction results. Some interactive plots are as follows:')
    st.markdown('---')
    
    df['color'] = df['price_range_predict'].apply(lambda x: 'khaki' if x == 0 else 'gold' if x == 1 else 'goldenrod' if x == 2 else 'darkgoldenrod')
    #0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'
    st.sidebar.subheader('Main Feature Selection')

    ############# Filters ######################################
    num_cores = st.sidebar.multiselect('* Select Number of Cores of Processor', ['All',1,2,3,4,5,6,7,8], [2,4])

    talk_time = st.sidebar.multiselect('* Select Number of Talk Time', ['All',2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], [9,10,11,12,13])
    st.sidebar.caption('_Longest Time A Single Battery Charge Will Last When You Are in h_')

    start1, end1 =st.sidebar.select_slider('* Select Random Access Memory Range', options = df.ram.drop_duplicates().sort_values(), value=(df.ram.min(), df.ram.max()))
    st.sidebar.caption('_Random Access Memory in Megabytes_') 
    
    start2, end2 =st.sidebar.select_slider("* Select Mobile Weight Range", options = df.mobile_wt.drop_duplicates().sort_values(), value=(df.mobile_wt.min(), df.mobile_wt.max()))
    st.sidebar.caption('_Total Weight of Mobile Phone in g_')

    start3, end3 =st.sidebar.select_slider("* Select Mobile Battery Power Rnage", options = df.battery_power.drop_duplicates().sort_values(), value=(df.battery_power.min(), df.battery_power.max()))
    st.sidebar.caption('_Battery Power of Mobile Phone in in mAh_')


    ######### Main Story Plot ####################################
    #count price_range_predict
    col1, col2 = st.columns((2,1))
    with col1: 
        ax = df.groupby(['price_range_predict']).size().reset_index(name='count').plot.pie(x='price_range_predict', y='count', autopct = '%0.2f%%' ,figsize=(12,6), colors = ("khaki", "gold", "goldenrod", 'darkgoldenrod'))
        ax.legend(title='price_range_predict', bbox_to_anchor=(1, 0.5), ncol=1)
        st.pyplot(ax.figure)
    with col2:
        st.subheader('The big story')
        st.write('**0: low cost**')
        st.write('**1: medium cost**')
        st.write('**2: high cost**')
        st.write('**3: very high cost**')
        st.write('**This bar plot shows that The mobile phone data of the four different prices is relatively balanced.**')
    st.markdown('---')


    #battery_power/fc/pc/clock_speed... vs price_range_predict
    col1, col2 = st.columns((2,1))
    with col1: 
        fig, axarr = plt.subplots(3, 2, figsize=(12, 6))
        ax=sns.boxplot(y='int_memory',x = 'price_range_predict', hue = 'price_range_predict',data = df, ax=axarr[0][0], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
        ax.legend([],[], frameon=False)
        ax=sns.boxplot(y='ram',x = 'price_range_predict', hue = 'price_range_predict',data = df, ax=axarr[0][1], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
        ax.legend([],[], frameon=False)
        ax=sns.boxplot(y='pc',x = 'price_range_predict', hue = 'price_range_predict',data = df, ax=axarr[1][0], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
        ax.legend([],[], frameon=False)
        ax=sns.boxplot(y='fc',x = 'price_range_predict', hue = 'price_range_predict',data = df, ax=axarr[1][1], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
        ax.legend([],[], frameon=False)
        ax=sns.boxplot(y='clock_speed',x = 'price_range_predict', hue = 'price_range_predict',data = df, ax=axarr[2][0], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
        ax.legend([],[], frameon=False)
        ax=sns.boxplot(y='battery_power',x = 'price_range_predict', hue = 'price_range_predict',data = df, ax=axarr[2][1], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
        ax.legend(loc='center left',title='price_range_predict', bbox_to_anchor=(1, 0.5), ncol=1)
        st.pyplot(ax.figure)
    with col2:
        st.write('**This boxplot shows: 1.The more expensive the phones, the higher ram and more battery power they tend to have; 2.At the same price level, pc is always higher than fc; 3.Clock speed and int_memory have no clear trend towards price range.**')
    st.markdown('---') 

    ######### Other Story Plot ####################################
    #n_cores vs price_range_predict
    col1, col2= st.columns((2,1))
    with col1: 
        if num_cores[0] != "All":
            df_filter1 = df.loc[df.n_cores.isin(num_cores), :]
            ax = pd.crosstab(df_filter1.n_cores, df_filter1.price_range_predict).plot(
                    kind="bar", 
                    figsize=(10,5),
                    xlabel = "Number of cores of processor",
                    color={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
            ax.legend(loc='center left', title='price_range_predict',bbox_to_anchor=(1, 0.5), ncol=1)
            st.pyplot(ax.figure)
        elif num_cores[0] == 'All':
            ax = pd.crosstab(df.n_cores, df.price_range_predict).plot(
                    kind="bar", 
                    figsize=(10,5),
                    xlabel = "Number of cores of processor",
                    color={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
            ax.legend(loc='center left', title='price_range_predict',bbox_to_anchor=(1, 0.5), ncol=1)
            st.pyplot(ax.figure)
        else:
            st.write('Please select again.')
    with col2:
        st.write('**This plot shows number of predicted price range for different number of processor cores.**')
    st.markdown('---') 

    #talk_time vs price_range_predict
    col1, col2= st.columns((2,1))
    with col1: 
        if talk_time[0] != "All":
            df_filter5 = df.loc[df.talk_time.isin(talk_time), :]
            ax = pd.crosstab(df_filter5.talk_time, df_filter5.price_range_predict).plot(
            kind="bar", 
            figsize=(10,5),
            xlabel = "Talk time",
            color={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
            ax.legend(loc='center left', title='price_range_predict',bbox_to_anchor=(1, 0.5), ncol=1)
            st.pyplot(ax.figure)
        elif talk_time[0] == 'All':
            ax = pd.crosstab(df.talk_time, df.price_range_predict).plot(
                    kind="bar", 
                    figsize=(10,5),
                    xlabel = "Number of Talk Time in hours",
                    color={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'})
            ax.legend(loc='center left', title='price_range_predict',bbox_to_anchor=(1, 0.5), ncol=1)
            st.pyplot(ax.figure)
        else:
            st.write('Please select again.')
    with col2:
        st.write('**This plot shows number of predicted price range for different talk time.**')
    st.markdown('---')       

    #ram vs price_range_predict
    col1, col2 = st.columns((2,1))
    with col1: 
        df_filter3 = df.loc[(df.ram >= start1) & (df.ram <= end1) , :]
        ax = df_filter3.plot.scatter(x='price_range_predict',y='ram',  c='color', figsize=(10,5))
        st.pyplot(ax.figure)
    with col2:
        st.write('**This plot shows how feature of random access memory distributes in different price range.**')
    st.markdown('---')  

    # pc vs fc
    col1, col2 = st.columns((2,1))
    with col1:
         df_filter1 = df.loc[df.n_cores.isin(num_cores), :]
         ax = df_filter1.plot.scatter(x='pc',y='fc',c='color', figsize=(10,5))
         st.pyplot(ax.figure)
    with col2:
        st.write('**This plot indicates that mobile phone with or without mega camera pixels actually does not affect the pricing a lot.**')
        st.caption('fc: Front Camera mega pixels')
        st.caption('pc: Primary Camera mega pixels')
    st.markdown('---')

    #int_memory/battery/talk_time/mobile_wt vs ram
    col1, col2 = st.columns((2,1))
    with col1: 
        fig, axarr = plt.subplots(2, 2, figsize=(8, 4))
        df_filter4 = df.loc[(df.ram >= start1) & (df.ram <= end1) & (df.battery_power >= start3) & (df.battery_power <= end3), :]
        ax=sns.scatterplot(y='battery_power',x = 'ram', hue = 'price_range_predict',data = df_filter4, ax=axarr[0][0], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'}, marker = "*")
        ax.legend([],[], frameon=False)

        df_filter3 = df.loc[(df.ram >= start1) & (df.ram <= end1) , :]
        ax=sns.scatterplot(y='int_memory',x = 'ram', hue = 'price_range_predict',data = df_filter3, ax=axarr[0][1], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'}, marker = "^")
        ax.legend([],[], frameon=False)

        df_filter2 = df.loc[(df.ram >= start1) & (df.ram <= end1) & (df.mobile_wt >= start2) & (df.mobile_wt <= end2), :]
        ax=sns.scatterplot(y='mobile_wt',x = 'ram', hue = 'price_range_predict',data = df_filter2, ax=axarr[1][0], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'}, marker = "^")
        ax.legend([],[], frameon=False)

        df_filter3 = df.loc[(df.ram >= start1) & (df.ram <= end1) , :]
        ax=sns.scatterplot(y='talk_time',x = 'ram', hue = 'price_range_predict',data = df_filter3, ax=axarr[1][1], palette={0:'khaki', 1: 'gold', 2:'goldenrod',3:'darkgoldenrod'}, marker = "*")
        
        ax.legend(loc='center left',title='price_range_predict', bbox_to_anchor=(1, 0.5), ncol=1)
        st.pyplot(ax.figure)
    with col2:
        st.write('**These scatter plots depicts the relationship between randam access memory(ram) versus battery power/internal meory/pc/talk time. Apparently, you can observe these points are clustered via the tints and shades;the darker the color, the more expensive it is.**')
    st.markdown('---') 

    #battery_power vs. talktime
    col1, col2 = st.columns((2,1))
    with col1:
         df_filter4 = df.loc[(df.battery_power >= start3) & (df.battery_power <= end3) & (df.mobile_wt >= start2) & (df.mobile_wt <= end2), :]
         ax = df_filter4.plot.scatter(x='mobile_wt',y='battery_power', c = 'talk_time', cmap='viridis', alpha = 0.5,  figsize=(12,6))
         st.pyplot(ax.figure)
    with col2:
        st.write('**This scatter plot reveals an interesting fact: the length of talk time does not directly relate to the battery power or mobile weight.**')
    st.markdown('---')

    #sample data show
    with st.expander('You can click here to see the sample data first 👇'):
        df_sample=df.sample(10)
        col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
        with col1:
            st.write('price_range')
        with col2:
            st.write('battery_power')
        with col3:
            st.write('clock_speed')
        with col4:
            st.write('fc')
        with col5:
            st.write('int_memory')
        with col6:
            st.write('mobile_wt')
        with col7:
            st.write('n_cores')
        with col8:
            st.write('pc')
        with col9:
            st.write('ram')
        with col10:
            st.write('talk_time')
        for index, row in df_sample.iterrows():
            col1, col2, col3, col4, col5, col6, col7, col8, col9, col10 = st.columns(10)
            with col1:
                if row['price_range_predict'] == 0:
                    st.error(0)
                elif row['price_range_predict'] == 1:
                    st.warning(1)
                elif row['price_range_predict'] == 2:
                    st.info(2)
                else:
                    st.success(3)
            with col2:
                st.write(row['battery_power'])
            with col3:
                st.write(row['clock_speed'])
            with col4:
                st.write(row['fc'])
            with col5:
                st.write(row['int_memory'])
            with col6:
                st.write(row['mobile_wt'])
            with col7:
                st.write(row['n_cores'])
            with col8:
                st.write(row['pc'])
            with col9:
                st.write(row['ram'])
            with col10:
                st.write(row['talk_time'])
     

################################################################
############### Model Training and Evaluation ##################
################################################################

elif page_selected == 'Model':
    # dataset info
    if st.button('About dataset'):
        st.write('The training dataset has 2000 rows and 10 columns. The feature columns contain 9 mobile phone performance such as battery power, blue, mobile weight, clock speed, talk time, wifi, etc., and the corresponding target column mobile price range 0, 1, 2, 3. ')
        st.write('The testing data which are used to do prediction of price range contains 1000 rows and 9 columns. The column of price_range in training dataset counts 500 values for 0, 1, 2, 3 respectively, so the dataset is pretty balanced, we do not need to do re-sampling. ')
        st.write('The belowing thermal map shows my data exploration of correlation coefficients.')
        st.image('df_corr.png')

    # model selection
    if st.button('About model building and evaluation'):
        st.write('I created pipeline to fill missing values with mode, and do standardization to keep data in the same scale, and try to add three kinds of classifer model to fit. The pipelines are as follows:')
        col1, col2, col3,col4= st.columns(4)
        col1.metric("Pipeline1(RandomForestClassifier)","79.5%")
        col2.metric("Pipeline2(DecisionTreeClassifier)","78%", "-1.9%")
        col3.metric("Pipeline3(LogisticRegression)", "81.5%", "4.5%")
        
        col7, col8 = st.columns([2,1])
        with col7:
            st.image('model.png')
        with col8:
            st.write('RandomForestClassifier performs accuracy as 79.5%, DecisionTreeClassifier perform accuracy as 78%, LogisticRegression performs accuracy as 81.5%. Thus, I choose to add LogisticRegression into the final pipeline.')
            st.write(' \n  \n ')
            st.write('**Final model traning accuracy:**',training_accuracy)
            st.write('**Confusion matrix:**', confusion_matrix)

    # interactive interface to do prediction
    st.write('  ')
    st.markdown(':iphone:'*77)
    col1, col2 ,col3= st.columns([2,0.1,1.6])
    with col1:
        st.subheader('Please set different parameters of mobile :iphone: to predict!')
        ram = st.number_input('Select Random Access Memory Range(M). (Recommend: 256~3998) ',0, 12288)
        battery_power = st.number_input('Select Battery Power Range(mAh). (Recommend: 500~2000)',500, 2000)
        clock_speed = st.number_input('Select Clock Speed Range(s). (Recommend: 0.5~3)',0.0, 3.0)
        fc = st.number_input('Select Front Camera Range(mega pixels). (Recommend: 0~20)',0, 20)
        pc = st.number_input('Select Primary Camera Range(mega pixels). (Recommend: 0~20)',0,20)
        int_memory = st.number_input('Select Internal Memory Range(G). (Recommend: 2~64)',0, 512)
        n_cores = st.selectbox('Select Number of Cores of Processor. (Recommend: 1~8)',[1,2,3,4,5,6,7,8])
        talk_time = st.number_input('Select Talk Time Range(h). (Recommend: 2~20)',2, 20)
        mobile_wt = st.number_input('Select Mobile Weight Range(g). (Recommend: 80~200)',80, 200)
        dic = {'battery_power':battery_power, 'clock_speed':clock_speed, 'fc':fc, 'int_memory':int_memory, 'mobile_wt':mobile_wt, 'n_cores':n_cores, 'pc':pc, 'ram':ram, 'talk_time':talk_time}
        ddd = pd.DataFrame(dic,index=[0])
        result = pipeline.predict(ddd)
    with col3:
        st.subheader('Here is your predicted price range!:sunglasses:')
        st.title(result)
        if result ==3:
            st.write("![3](https://media.giphy.com/media/JUqiFbumTAPYIeM8yJ/giphy.gif)")
        
################################################################
############### About Page ##################
################################################################# 
else: 
    st.subheader('We are Team4.')
    st.snow()
    col1, col2 = st.columns(2)
    with col1:
        st.write("![Your Awsome GIF](https://media.giphy.com/media/txsJLp7Z8zAic/giphy.gif)")
    with col2:
        st.write("![Your Awsome GIF](https://media.giphy.com/media/txsJLp7Z8zAic/giphy.gif)")
    

      
  

