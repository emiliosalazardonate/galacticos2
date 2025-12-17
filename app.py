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
    st.title('Finetuning Zoobot Software')
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
    questions = {
        'bar': ['strong', 'weak', 'no'],
        'has-spiral-arms': ['yes', 'no'],
        'spiral-arm-count': ['1', '2', '3', '4'],
        'spiral-winding': ['tight', 'medium', 'loose'],
        'merging': ['merger', 'major-disturbance', 'minor-disturbance', 'none'],
        'disk-edge-on_yes_fraction':  ['yes', 'no'],
    }
    # could make merging yes/no

    # st.sidebar.markdown('# Show posteriors')
    # show_posteriors = st.sidebar.selectbox('Posteriors for which question?', ['none'] + list(questions.keys()), format_func=lambda x: x.replace('-', ' ').capitalize())

    st.sidebar.markdown('# Choose Your Galaxies')
    # st.sidebar.markdown('---')
    current_selection = {}
    for question, answers in questions.items():
        valid_to_select = True
        st.sidebar.markdown("# " + question.replace('-', ' ').capitalize() + '?')

        # control valid_to_select depending on if question is relevant
        if question.startswith('spiral-'):
            has_spiral_answer, has_spiral_mean = current_selection.get('has-spiral-arms', [None, None])
            # logging.info(f'has_spiral limits: {has_spiral_mean}')
            if has_spiral_answer == 'yes':
                valid_to_select = np.min(has_spiral_mean) > 0.5
            else:
                valid_to_select = np.min(has_spiral_mean) < 0.5

        if valid_to_select:
            selected_answer = st.sidebar.selectbox('Answer', answers,
                                                   format_func=lambda x: x.replace('-', ' ').capitalize(),
                                                   key=question + '_select')
            selected_mean = st.sidebar.slider(
                label='Posterior Mean',
                value=[.0, 1.],
                key=question + '_mean')
            current_selection[question] = (selected_answer, selected_mean)
            # and sort by confidence, for now
        else:
            st.sidebar.markdown('*To use this filter, set "Has Spiral Arms = Yes"to > 0.5*'.format(question))
            current_selection[question] = None, None

    galaxies = df
    logging.info('Total galaxies: {}'.format(len(galaxies)))
    valid = np.ones(len(df)).astype(bool)
    for question, answers in questions.items():
        answer, mean = current_selection.get(question, [None, None])  # mean is (min, max) limits
        logging.info(f'Current: {question}, {answer}, {mean}')
        if mean == None:  # happens when spiral count question not relevant
            mean = (None, None)
        if len(mean) == 1:
            # streamlit sharing bug is giving only the higher value
            logging.info('Streamlit bug is happening, working')
            mean = (0., mean[0])
        # st.markdown('{} {} {} {}'.format(question, answers, answer, mean))
        if (answer is not None) and (mean is not None):
            # this_answer = galaxies[question + '_' + answer + '_concentration_mean']
            # all_answers = galaxies[[question + '_' + a + '_concentration_mean' for a in answers]].sum(axis=1)
            this_answer = galaxies[question + '_' + answer + '_fraction']
            all_answers = galaxies[[question + '_' + a + '_fraction' for a in answers]].sum(axis=1)
            prob = this_answer / all_answers
            within_limits = (np.min(mean) <= prob) & (prob <= np.max(mean))

            preceding = True
            if mean != (0., 1.):
                preceding = galaxies[question + '_proportion_volunteers_asked'] >= 0.5

            logging.info('Fraction of galaxies within limits: {}'.format(within_limits.mean()))
            valid = valid & within_limits & preceding

    logging.info('Valid galaxies: {}'.format(valid.sum()))
    st.markdown('{:,} of {:,} galaxies match your criteria.'.format(valid.sum(), len(valid)))

    # selected = galaxies[valid].sample(np.min([valid.sum(), 16]))

    # image_locs = [row['file_loc'].replace('/decals/png_native', '/galaxy_zoo/gz2') for _, row in selected.iterrows()]
    # images = [np.array(Image.open(loc)).astype(np.uint8) for loc in image_locs]

    # if show_posteriors is not 'none':
    #     selected = galaxies[valid][:8]
    #     question = show_posteriors
    #     if question == 'spiral-count' or question == 'spiral-winding':
    #         st.markdown('Sorry! You asked to see posteriors for "{}", but this demo app only supports visualing posteriors for questions with two answers. Please choose another option.'.format(question.capitalize().replace('-', ' ')))
    #     else:
    #         answers = questions[question]
    #         selected_answer = current_selection[question][0]
    #         for _, galaxy in selected.iterrows():
    #             show_predictions(galaxy, question, answers, selected_answer)
    # else:
    # image_urls = ["https://panoptes-uploads.zooniverse.org/production/subject_location/02a32231-11c6-45b6-b448-fd85ec32fbd8.png"] * 16
    selected = galaxies[valid][:40]
    image_urls = selected['url']

    opening_html = '<div style=display:flex;flex-wrap:wrap>'
    closing_html = '</div>'
    child_html = ['<img src="{}" style=margin:3px;width:200px;></img>'.format(url) for url in image_urls]

    gallery_html = opening_html
    for child in child_html:
        gallery_html += child
    gallery_html += closing_html

    # st.markdown(gallery_html)
    st.markdown(gallery_html, unsafe_allow_html=True)



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
    df_locs = [script_dir / f'decals_{n}.csv' for n in range(4)]
    
    dfs = [pd.read_csv(df_loc) for df_loc in df_locs]
    print("loaded _data.......")
    return pd.concat(dfs)


if __name__ == '__main__':
    logging.basicConfig(level=logging.CRITICAL)

    df = load_data()
    main(df)
