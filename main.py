import streamlit as st
import pandas as pd
import altair as alt
from classifier import Classifier
from PIL import Image


def print_results():
    preds = classifier.get_result(user_input)
    if preds[0]:
        preds_dict = preds[1]
        info_text = "This comment is toxic! We are **" + str(int(preds_dict['toxic'] * 100)) + "%** sure"
        st.info(info_text)
        add_str = []
        preds_dict.pop('toxic', None)
        if any(preds_dict.values()) > 0.1:
            for key, value in preds_dict.items():
                if value > 0.1:
                    add_str.append("**" + key + "** (" + str(int(value * 100)) + "%)")
        if add_str:
            st.write("Also it could be marked as ", ', '.join(add_str))
    else:
        st.warning("This comment seems to be non-toxic.")


def show_graph():
    stats = pd.read_csv('target_stat.csv')
    c = alt.Chart(stats).mark_bar().encode(
        x="label",
        y="count"
    ).configure_axisX(labelAngle=0)
    st.altair_chart(c, use_container_width=True)


if __name__ == "__main__":
    classifier = Classifier()
    st.set_page_config("Toxic App", ":knife:")
    st.image(Image.open('toxic-website.png'), use_column_width=True)
    st.title('Toxic App :knife: :knife:')
    user_input = st.text_area("Input your comment here (english, please)")
    print_results()

    with st.beta_expander("Want to know more about the data?"):
        st.markdown('Dataset was taken from [Kaggle]'
                    ' (https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/data)')
        st.write('It consisted of more than 150 thousand user comments from wikipedia.')
        st.write('Here you can see the distribution of target variable')
        show_graph()
