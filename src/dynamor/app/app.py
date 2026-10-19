import streamlit as st

from dynamor.services.recommender import group_items, load_context, suggest, validate_draft
from dynamor.settings import settings

HEADINGS = (
    ("weapon", "Weapon"),
    ("vitality", "Vitality"),
    ("spirit", "Spirit"),
)

st.set_page_config(page_title="DynamoR", layout="wide")
st.title("DynamoR")

@st.cache_resource
def cached_context():
    return load_context(settings)

try:
    context = cached_context()
except FileNotFoundError:
    st.error("Run python scripts/build_index.py first")
    st.stop()

heroes = sorted(context.heroes, key=lambda hero: (hero["name"], hero["id"]))
labels = {hero["id"]: hero["name"] for hero in heroes}
hero_ids = [hero["id"] for hero in heroes]
own_id = st.selectbox("Your hero", hero_ids, format_func=labels.get)
enemy_ids = []
for number in range(1, 7):
    picked = st.selectbox(
        f"Enemy {number}",
        [None, *hero_ids],
        format_func=lambda hero_id: "" if hero_id is None else labels[hero_id],
        key=f"enemy-{number}",
    )
    enemy_ids.append(picked)

message = validate_draft(own_id, enemy_ids, hero_ids)
if message is not None:
    st.warning(message)
    st.stop()

shown = suggest(
    own_id,
    enemy_ids,
    context.index,
    context.targets,
    context.hero_ids,
    context.hero_catalog,
    context.item_catalog,
    context.names,
    settings.top_k,
    settings.top_items,
)
if not shown:
    st.info("No winning drafts for this hero")
    st.stop()

grouped = group_items(shown)
columns = st.columns(3)
for column, (category, heading) in zip(columns, HEADINGS):
    column.subheader(heading)
    rows = grouped[category]
    if not rows:
        column.caption("No items in this category")
    for item in rows:
        column.write(f"{item.name} — {item.confidence_pct}%")

if grouped["other"]:
    st.subheader("Other")
    for item in grouped["other"]:
        st.write(f"{item.name} — {item.confidence_pct}%")