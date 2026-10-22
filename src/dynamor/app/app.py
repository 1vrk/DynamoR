import html
from datetime import datetime, timezone

import streamlit as st

from dynamor.services.recommender import (
    INVEST_CATEGORIES,
    INVEST_TARGET,
    group_phases,
    investment_progress,
    load_context,
    suggest,
    validate_draft,
)
from dynamor.services.refresh import (
    STAGES,
    RefreshError,
    StageProgress,
    read_refresh_stamp,
    refresh_is_due,
    run_refresh,
)
from dynamor.settings import settings

PHASES = (
    ("early", "Early"),
    ("mid", "Mid"),
    ("late", "Late"),
)
BORDERS = {
    "weapon": "#E0863A",
    "vitality": "#3DDC84",
    "spirit": "#B46CFF",
    "skill": "#E6E6E6",
    "other": "#8A8A8A",
}
INVEST_LABELS = {
    "weapon": "Weapon",
    "vitality": "Vitality",
    "spirit": "Spirit",
}


def _phase_html(items, *, show_time: bool, names=None, crossed=None) -> str:
    crossed = crossed or {}
    cards = []
    for item in items:
        border = BORDERS.get(item.category, BORDERS["other"])
        picture = ""
        if item.image:
            picture = (
                f'<img src="{html.escape(item.image)}" alt="" width="64" height="64" '
                'style="object-fit:contain">'
            )
        clock = ""
        if show_time and item.purchase_s is not None:
            clock = f"<div>{item.purchase_s // 60}:{item.purchase_s % 60:02d}</div>"
        meta = [str(item.cost)] if item.cost is not None else []
        meta.append(f"{item.confidence_pct}%")
        upgrade = ""
        if item.upgrade_from is not None and names is not None:
            parent = names.get(item.upgrade_from)
            parent_name = parent[0] if parent is not None else str(item.upgrade_from)
            upgrade = f"<div>Усиление · {html.escape(parent_name)}</div>"
        closed = "<div>4800</div>" if item.item_id in crossed else ""
        cards.append(
            f'<div style="border:2px solid {border};border-radius:8px;padding:8px;'
            f'width:120px;text-align:center">'
            f"{picture}"
            f'<div style="font-weight:600">{html.escape(item.name)}</div>'
            f"{upgrade}"
            f"{closed}"
            f"{clock}"
            f'<div style="opacity:0.75;font-size:0.85em">{html.escape(" · ".join(meta))}</div>'
            f"</div>"
        )
    return '<div style="display:flex;gap:8px;flex-wrap:wrap">' + "".join(cards) + "</div>"


def _stage_line(progress: StageProgress) -> str:
    if progress.state == "done":
        return f"{progress.stage}: Done"
    if progress.state == "running":
        return f"{progress.stage}: Running"
    if progress.state == "failed":
        return f"{progress.stage}: {progress.error}"
    return f"{progress.stage}: Waiting"


def _render_data() -> None:
    finished = read_refresh_stamp(settings)
    if finished is None:
        st.caption("No update yet")
    else:
        st.caption(f"Last update: {finished.isoformat()}")
        if refresh_is_due(finished, now=datetime.now(timezone.utc)):
            st.warning("Last update is older than 7 days.")

    bar = st.progress(0.0)
    slots = {stage: st.empty() for stage in STAGES}
    states = {
        stage: StageProgress(stage, "waiting")
        for stage in STAGES
    }

    def paint() -> None:
        done = sum(1 for stage in STAGES if states[stage].state == "done")
        running = next((states[stage] for stage in STAGES if states[stage].state == "running"), None)
        fraction = 0.0 if running is None else running.fraction
        bar.progress(min(1.0, (done + fraction) / len(STAGES)))
        for stage in STAGES:
            slots[stage].write(_stage_line(states[stage]))

    paint()
    if st.button("Update data"):
        def on_progress(progress: StageProgress) -> None:
            states[progress.stage] = progress
            paint()

        try:
            run_refresh(settings, on_progress)
        except RefreshError as exc:
            st.error(str(exc))
        else:
            st.cache_resource.clear()
            st.rerun()


def _render_build() -> None:
    try:
        context = cached_context()
    except FileNotFoundError:
        st.error("Run python scripts/build_index.py first")
        return

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
        return

    build = suggest(
        own_id,
        enemy_ids,
        context.index,
        context.targets,
        context.hero_ids,
        context.hero_catalog,
        context.item_catalog,
        context.names,
        context.item_times,
        settings.top_k,
        settings.top_items,
        context.skills.get(own_id, []),
        context.upgrades,
        context.purchases,
    )
    if not build.items and not build.skills:
        st.info("No winning drafts for this hero")
        return

    st.subheader("Skills")
    if not build.skills:
        st.caption("No items")
    else:
        st.markdown(_phase_html(build.skills, show_time=False), unsafe_allow_html=True)

    close = context.scales.get(own_id, ())
    progress = investment_progress(build.items, close)
    for category in INVEST_CATEGORIES:
        label = INVEST_LABELS[category]
        total = progress.totals[category]
        if category in close:
            st.write(f"{label} {total} / {INVEST_TARGET}")
        else:
            st.write(f"{label} {total}")

    phases = group_phases(build.items)
    for key, heading in PHASES:
        st.subheader(heading)
        rows = phases[key]
        if not rows:
            st.caption("No items")
        else:
            st.markdown(
                _phase_html(rows, show_time=True, names=context.names, crossed=progress.crossed),
                unsafe_allow_html=True,
            )


st.set_page_config(page_title="DynamoR", layout="wide")
st.title("DynamoR")


@st.cache_resource
def cached_context():
    return load_context(settings)


build_tab, data_tab = st.tabs(["Build", "Data"])
with data_tab:
    _render_data()
with build_tab:
    _render_build()
