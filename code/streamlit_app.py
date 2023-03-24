import os
import streamlit as st
from pos_tagger import load_dataset, get_matrices, preprocess_to_infer, inference


st.set_page_config(page_title="HMM Viterbi PoS-Tagging", layout="centered")


@st.cache_resource
def get_pretty_resource():
    TPL_ENT = '<mark class="entity" style="background: {bg}; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">{text}<span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; vertical-align: middle; margin-left: 0.5rem">{tag}</span></mark>'
    TPL_ENTS = '<div class="entities" style="line-height: 2.5; direction: ltr">{}</div>'
    WRAPPER = '<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem; margin-bottom: 2.5rem">{}</div>'
    style = "<style>mark.entity { display: inline-block }</style>"
    colors = [
        "#7aecec",
        "#bfeeb7",
        "#feca74",
        "#ff9561",
        "#aa9cfc",
        "#c887fb",
        "#9cc9cc",
        "#ffeb80",
        "#ff8197",
        "#ff8197",
        "#f0d0ff",
        "#bfe1d9",
        "#bfe1d9",
        "#e4e7d2",
        "#e4e7d2",
        "#e4e7d2",
        "#e4e7d2",
        "#e4e7d2",
    ]

    return {
        "TPL_ENT": TPL_ENT,
        "TPL_ENTS": TPL_ENTS,
        "WRAPPER": WRAPPER,
        "style": style,
        "colors": colors,
    }


def prettify(tagged_text: list[tuple[str, str]], tagset):
    """Convert HTML so it can be rendered."""

    pretty_resource = get_pretty_resource()

    html = ""
    for text, tag in tagged_text:
        color = pretty_resource["colors"][
            tagset.index(tag) % len(pretty_resource["colors"])
        ]
        html += pretty_resource["TPL_ENT"].format(text=text, tag=tag, bg=color)

    html = pretty_resource["TPL_ENTS"].format(html)

    # Newlines seem to mess with the rendering
    html = html.replace("\n", " ")

    return f'{pretty_resource["style"]}{pretty_resource["WRAPPER"].format(html)}'


@st.cache_resource
def get_cached_matrices(language):
    if language == "Portuguese":
        train_filepath = os.path.join("..", "data", "macmorpho", "macmorpho-train.txt")
    elif language == "English":
        train_filepath = os.path.join("..", "data", "WSJ", "WSJ_02-21.txt")
    else:
        raise ValueError("Language not defined.")

    dataset = load_dataset(train_filepath)
    transition_matrix, emission_matrix, tagset, vocab_idx, lower_vocab = get_matrices(
        dataset
    )

    return transition_matrix, emission_matrix, tagset, vocab_idx, lower_vocab


def main():
    st.title("HMM Viterbi PoS-Tagging")

    language = st.radio(
        "Language",
        ("English", "Portuguese"),
        help="English: Trained on WSJ.\n\nPortuguese: Trained on Mac-Morpho.\n\nThe tagset is respective to its training dataset.",
    )

    (
        transition_matrix,
        emission_matrix,
        tagset,
        vocab_idx,
        lower_vocab,
    ) = get_cached_matrices(language)

    text = st.text_area(label="Input text")
    clicked = st.button("PoS tag!")

    if clicked:
        text, input_text = preprocess_to_infer(text, vocab_idx, lower_vocab)
        Y_hat = inference(
            input_text, transition_matrix, emission_matrix, tagset, vocab_idx
        )

        st.write(prettify(zip(text, Y_hat), tagset), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
