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
            upgrade = f"<div>Upgrade · {html.escape(parent_name)}</div>"
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


def _counter_html(counters) -> str:
    cards = []
    for counter in counters:
        border = BORDERS.get(counter.category, BORDERS["other"])
        picture = ""
        if counter.image:
            picture = (
                f'<img src="{html.escape(counter.image)}" alt="" width="64" height="64" '
                'style="object-fit:contain">'
            )
        note = "<div>already in queue</div>" if counter.already else ""
        if counter.replaces:
            note = f"<div>instead of {html.escape(counter.replaces)}</div>"
        cost = "" if counter.cost is None else f"<div>{counter.cost}</div>"
        cards.append(
            '<div style="border:2px solid '
            f'{border};border-radius:8px;padding:8px;width:160px;text-align:center">'
            f"{picture}"
            f'<div style="font-weight:600">{html.escape(counter.name)}</div>'
            f'<div style="font-size:0.85em">{html.escape(counter.reason)}</div>'
            f"{note}{cost}"
            "</div>"
        )
    return '<div style="display:flex;gap:8px;flex-wrap:wrap">' + "".join(cards) + "</div>"


def _chart_html(points) -> str:
    span = 30 * 60
    marks = []
    for point in points:
        left = min(point.purchase_s, span) / span * 100
        top = (1 - max(0.0, min(1.0, point.keep_share))) * 100
        border = BORDERS.get(point.category, BORDERS["other"])
        thickness = 3 if point.answers else 1
        picture = ""
        if point.image:
            picture = (
                f'<img src="{html.escape(point.image)}" alt="" width="28" height="28" '
                'style="object-fit:contain">'
            )
        title = (
            f"{point.name} {point.purchase_s // 60}:{point.purchase_s % 60:02d} "
            f"{round(point.keep_share * 100)}%"
        )
        marks.append(
            f'<div title="{html.escape(title)}" style="position:absolute;left:{left:.2f}%;'
            f"top:{top:.2f}%;transform:translate(-50%,-50%);width:36px;height:36px;"
            f"border:{thickness}px solid {border};border-radius:6px;background:#241f1a;"
            f'display:flex;align-items:center;justify-content:center;overflow:hidden">'
            f"{picture}</div>"
        )
    plot = (
        '<div style="position:absolute;left:8px;right:8px;top:32px;bottom:28px">'
        '<div style="position:absolute;left:33.3%;top:0;bottom:0;border-left:1px dashed #666"></div>'
        '<div style="position:absolute;left:66.6%;top:0;bottom:0;border-left:1px dashed #666"></div>'
        '<div style="position:absolute;left:0;right:0;top:50%;border-top:1px dashed #444"></div>'
        + "".join(marks)
        + "</div>"
    )
    return (
        '<div style="position:relative;height:340px;background:#1b1a17;border-radius:8px">'
        '<div style="position:absolute;left:8px;top:8px;font-size:12px;opacity:0.75">Strong Early</div>'
        '<div style="position:absolute;right:8px;top:8px;font-size:12px;opacity:0.75">Strong Late</div>'
        '<div style="position:absolute;left:8px;bottom:32px;font-size:12px;opacity:0.75">Weak Early</div>'
        '<div style="position:absolute;right:8px;bottom:32px;font-size:12px;opacity:0.75">Weak Late</div>'
        + plot
        + '<div style="position:absolute;left:33.3%;bottom:4px;transform:translateX(-50%);font-size:12px">10m</div>'
        '<div style="position:absolute;left:66.6%;bottom:4px;transform:translateX(-50%);font-size:12px">20m</div>'
        '<div style="position:absolute;right:8px;bottom:4px;font-size:12px">30m</div>'
        "</div>"
    )


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
        context.scale_weights.get(own_id, {}),
        context.item_stats,
        context.hero_tags,
        context.item_answers,
        context.item_effects,
        labels,
    )
    if not build.items and not build.skills and not build.counters and not build.points:
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

    st.subheader("Counters")
    if not build.counters:
        st.caption("No items")
    else:
        st.markdown(_counter_html(build.counters), unsafe_allow_html=True)

    st.subheader("Chart")
    if not build.points:
        st.caption("No items")
    else:
        st.markdown(_chart_html(build.points), unsafe_allow_html=True)


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
