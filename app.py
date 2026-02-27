import json
import logging

import streamlit as st
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
# import tensorflow as tf
# import tensorflow_probability as tfp
from PIL import Image


def main(df):
    print ("main.......")
    st.title('Finetuning Zoobot Software: Emilio Salazar')
    st.subheader('')

    st.markdown(
        """
    
        <br><br/>
        Galaxy Zoo DECaLS includes deep learning classifications for all galaxies. 
    
        Our model learns from volunteers and predicts posteriors for every Galaxy Zoo question.
    
        Explore the predictions using the filters on the left. Do you agree with the model?
    
        To read more about how the model works, click below.
    
        """
        , unsafe_allow_html=True)
    should_tell_me_more = st.button('Tell me more')
    if should_tell_me_more:
        tell_me_more()
        st.markdown('---')
    else:
        st.markdown('---')
        interactive_galaxies(df)


def tell_me_more():
    st.title('Building the Model')

    st.button('Back to galaxies')  # will change state and hence trigger rerun and hence reset should_tell_me_more

    st.markdown("""
    We require a model which can:
    - Learn efficiently from volunteer responses of varying (i.e. heteroskedastic) uncertainty
    - Predict posteriors for those responses on new galaxies, for every question

    In [previous work](https://arxiv.org/abs/1905.07424), we modelled volunteer responses as being binomially distributed and trained our model to make maximum likelihood estimates using the loss function:
    """)

    st.latex(
        """
        \mathcal{L} = k \log f^w(x) + (N-k) \log(1-f^w(x))
        """
    )
    st.markdown(
        r"""
        where, for some target question, k is the number of responses (successes) of some target answer, N is the total number of responses (trials) to all answers, and $f^w(x) = \hat{\rho}$ is the predicted probability of a volunteer giving that answer.
        """
    )

    st.markdown(
        r"""
        This binomial assumption, while broadly successful, broke down for galaxies with vote fractions k/N close to 0 or 1, where the Binomial likelihood is extremely sensitive to $f^w(x)$, and for galaxies where the question asked was not appropriate (e.g. predict if a featureless galaxy has a bar). 
    
        Instead, in our latest work, the model predicts a distribution 
        """)

    st.latex(r"""
    f^w(x) = p(\rho|f^w(x))
    """)

    st.markdown(r"""
    and $\rho$ is then drawn from that distribution.

    For binary questions, one could use the Beta distribution (being flexible and defined on the unit interval), and predict the Beta distribution parameters $f^w(x) = (\hat{\alpha}, \hat{\beta})$ by minimising

    """)

    st.latex(
        r"""
            \mathcal{L} = \int Bin(k|\rho, N) Beta(\rho|\alpha, \beta) d\alpha d\beta    
        """
    )
    st.markdown(r"""

    where the Binomial and Beta distributions are conjugate and hence this integral can be evaluated analytically.

    In practice, we would like to predict the responses to questions with more than two answers, and hence we replace each distribution with its multivariate counterpart; Beta($\rho|\alpha, \beta$) with Dirichlet($\vec{\rho}|\vec{\alpha})$, and Binomial($k|\rho, N$) with Multinomial($\vec{k}|\vec{\rho}, N$).
    """)

    st.latex(r"""
     \mathcal{L}_q = \int Multi(\vec{k}|\vec{\rho}, N) Dirichlet(\vec{\rho}| \vec{\alpha}) d\vec{\alpha}
    """)

    st.markdown(r"""
    where $\vec{k}, \vec{\rho}$ and $\vec{\alpha}$ are now all vectors with one element per answer. 

    Using this loss function, our model can predict posteriors with excellent calibration.

    For the final GZ DECaLS predictions, I actually use an ensemble of models, and apply active learning - picking the galaxies where the models confidently disagree - to choose the most informative galaxies to label with Galaxy Zoo. Check out the paper for more.

    """)

    st.button('Back to galaxies',
              key='back_again')  # will change state and hence trigger rerun and hence reset should_tell_me_more


def interactive_galaxies(df):
    st.sidebar.markdown('# Filter Galaxies')

    # Extract all unique classification labels
    all_labels = df['classification_label'].unique().tolist()
    all_labels.sort() # Sort labels alphabetically

    # Add a selectbox for classification labels
    selected_label = st.sidebar.selectbox(
        'Select Classification Label',
        ['All'] + all_labels
    )

    filtered_df = df
    if selected_label != 'All':
        filtered_df = df[df['classification_label'] == selected_label]

    st.markdown(f'{len(filtered_df)} galaxies match your criteria.')

    # Pagination
    images_per_page = 30
    if 'num_images' not in st.session_state:
        st.session_state.num_images = images_per_page

    # Reset pagination when filter changes
    if 'last_selected_label' not in st.session_state or st.session_state.last_selected_label != selected_label:
        st.session_state.num_images = images_per_page
        st.session_state.last_selected_label = selected_label

    total_images = len(filtered_df)
    current_images = filtered_df.iloc[:st.session_state.num_images]

    if not current_images.empty:
        opening_html = '<div style=display:flex;flex-wrap:wrap>'
        closing_html = '</div>'
        child_html = [f'<div style="margin:5px; text-align:center"><img src="{row["image_url"]}" style="width:200px;"><br>{row["classification_label"]}</div>' for _, row in current_images.iterrows()]

        gallery_html = opening_html
        for child in child_html:
            gallery_html += child
        gallery_html += closing_html

        st.markdown(gallery_html, unsafe_allow_html=True)
    else:
        st.write("No images to display for the current selection.")

    # "Load More" button
    if st.session_state.num_images < total_images:
        if st.button('Load More'):
            st.session_state.num_images += images_per_page
            try:
                st.experimental_rerun()
            except AttributeError:
                st.rerun()


st.set_page_config(
    layout="wide",
    page_title='GZ DECaLS',
    page_icon='gz_icon.jpeg'
)


@st.cache
def load_data():
    print ("load_data.......")
    
    # Get the absolute path of the directory containing this script
    script_dir = Path(__file__).parent.absolute()
    
    # Load the classifications file
    classifications_df = pd.read_csv(script_dir / 'edge-on-galaxies-classifications1000.csv')
    
    # Load the subjects file
    subjects_df = pd.read_csv(script_dir / 'edge-on-galaxies-subjects.csv')
    
    # Merge on subject_id
    # classifications_df has 'subject_ids' (plural) and subjects_df has 'subject_id' (singular)
    # Assuming subject_ids in classifications_df is a single ID
    df = pd.merge(classifications_df, subjects_df, left_on='subject_ids', right_on='subject_id')

    # Parse 'annotations' and 'locations'
    def extract_info(row):
        try:
            annotations = json.loads(row['annotations'])
            # The structure is [{"task": "T0", "value": "Label"}] or [{"task": "T0", "value": ["Label"]}]
            if annotations and annotations[0]['value']:
                val = annotations[0]['value']
                if isinstance(val, list):
                    classification_label = val[0]
                else:
                    classification_label = val
            else:
                classification_label = 'Unknown'
        except (json.JSONDecodeError, IndexError, KeyError, TypeError):
            classification_label = 'Unknown'
        
        try:
            locations = json.loads(row['locations'])
            # The locations is a dict like {"0": "url"}
            image_url = list(locations.values())[0] if locations else ''
        except (json.JSONDecodeError, IndexError, KeyError, AttributeError):
            image_url = ''
        
        return classification_label, image_url

    # Apply the function to create new columns
    temp_df = df.apply(extract_info, axis=1, result_type='expand')
    df['classification_label'] = temp_df[0]
    df['image_url'] = temp_df[1]
    
    print("loaded _data.......")
    return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)

    df = load_data()
    main(df)
