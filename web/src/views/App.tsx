/** Composition only.
 *
 *  Every piece of state lives in a hook under `../controllers`, every rule
 *  about what a value set *is* lives in `../models/paramState`, and every
 *  pixel is drawn by a component under `./panels`, `./stage` or `./controls`.
 *  What is left here is the wiring: which hook feeds which panel.
 *
 *  The one thing worth reading for is the order the hooks run in. `useSchema`
 *  boots, `useValues` seeds itself from the schema the moment it lands, and
 *  everything downstream reads from those two -- so a hook that needs a value
 *  has to come after the hook that owns it.
 */

import { useMemo, useState } from "react";

import { useBeforePeek } from "../controllers/useBeforePeek";
import { useExport } from "../controllers/useExport";
import { useLuts } from "../controllers/useLuts";
import { usePresetFile } from "../controllers/usePresetFile";
import { usePreview } from "../controllers/usePreview";
import { useSchema } from "../controllers/useSchema";
import { useUpload } from "../controllers/useUpload";
import { useValues } from "../controllers/useValues";
import {
  BOARD_LIGHT_DEFAULT,
  BOARD_LIGHT_MAX,
  sectionDomId,
} from "../models/constants";
import { groupedParams, isNeutral } from "../models/paramState";
import type { Compare } from "../models/types";
import type { ImageMeta } from "../services/api";
import filmGrain16x9 from "../assets/film-grain-16x9.jpg";
import Field from "./controls/Field";
import ExportPanel from "./panels/ExportPanel";
import PresetPicker from "./panels/PresetPicker";
import ScalePanel from "./panels/ScalePanel";
import SectionMenu from "./panels/SectionMenu";
import SliderPanel from "./panels/SliderPanel";
import TopBar from "./panels/TopBar";
import Stage from "./stage/Stage";

export default function App() {
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [meta, setMeta] = useState<ImageMeta | null>(null);

  const [supersample, setSupersample] = useState(2);
  const [compare, setCompare] = useState<Compare>("overlay");
  const [split, setSplit] = useState(1); // 1 = fully processed
  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({});
  const [scaleToRef, setScaleToRef] = useState(true);
  /** Hand-set linear size-scaling factor, or null for the computed one. Here
   *  rather than in `useValues` because it is not part of the look: it is not
   *  saved into a preset file and it does not travel with one. */
  const [scaleOverride, setScaleOverride] = useState<number | null>(null);
  const [bgLightness, setBgLightness] = useState(BOARD_LIGHT_DEFAULT);

  /** Reroll `seed` and `texture_seed` whenever a photo is opened, so different
   *  photos do not render with the identical grain and damage pattern -- both
   *  are deterministic by design (that is what makes them re-orderable at
   *  all), so left alone they are the *same* deterministic pattern on every
   *  photo. On by default: a repeated seed is the surprising outcome. */
  const [randomizeSeedOnOpen, setRandomizeSeedOnOpen] = useState(true);

  const { schema, device, setDevice, luts, setLuts, booted } = useSchema(setError);
  // The photo's id is passed in so the edit history starts over when a
  // different one is opened -- a step describing a render that no longer
  // exists is not somewhere you can go back to.
  const v = useValues(schema, booted, meta?.id ?? null);
  const { showBefore, setShowBefore } = useBeforePeek();

  /** The reference size the *render* uses, which is not always the one the
   *  preset recorded. The server scales lengths by sqrt(photoMP / referenceMP),
   *  so a hand-set linear factor `f` is exactly the reference size that solves
   *  that for `f` -- which keeps the override on this side of the API and
   *  leaves `v.referenceMp` holding what the preset actually says, so saving a
   *  file still stamps the truth. */
  const renderReferenceMp =
    scaleOverride !== null && meta
      ? meta.megapixels / (scaleOverride * scaleOverride)
      : v.referenceMp;

  const preview = usePreview({
    meta,
    applied: v.applied,
    supersample,
    referenceMp: renderReferenceMp,
    scaleToRef,
    lut: v.lut,
    onError: setError,
    onDevice: setDevice,
  });

  const exporter = useExport({
    meta,
    values: v.values,
    supersample,
    referenceMp: renderReferenceMp,
    scaleToRef,
    lut: v.lut,
    onError: setError,
  });

  const upload = useUpload({
    onAccepted: () => {
      preview.clear();
      exporter.setJob(null);
    },
    onMeta: (m) => {
      setMeta(m);
      if (randomizeSeedOnOpen) v.randomizeSeeds();
    },
    onError: setError,
  });

  const lutCtl = useLuts({
    luts,
    setLuts,
    setLut: v.setLut,
    valuesRef: v.valuesRef,
    setValueNow: v.setValueNow,
    liveFor: v.liveFor,
    onError: setError,
    onNotice: setNotice,
  });

  const presetFile = usePresetFile({
    schema,
    values: v.values,
    meta,
    referenceMp: v.referenceMp,
    lut: v.lut,
    setReferenceMp: v.setReferenceMp,
    setLut: v.setLut,
    applyValues: v.applyValues,
    onError: setError,
    onNotice: setNotice,
  });

  const grouped = useMemo(() => groupedParams(schema), [schema]);
  // "Collapse all" while anything is open, "Expand all" once everything is
  // shut -- one button, and which way it goes is never in doubt because the
  // panel in front of you is the state it is reading.
  const allCollapsed = grouped.length > 0 && grouped.every((g) => collapsed[g.group]);
  const toggleAllCollapsed = () => {
    const next: Record<string, boolean> = {};
    for (const g of grouped) next[g.group] = !allCollapsed;
    setCollapsed(next);
  };

  /** Scroll the panel to a section, from the jump menu on the export bar.
   *
   *  A folded section is opened on the way: a jump that lands on a collapsed
   *  header shows nothing of what was jumped to, which reads as the menu not
   *  having worked. The scroll waits a frame because the expansion changes the
   *  panel's layout, and `scrollIntoView` measured before that re-render aims
   *  at where the section used to be. */
  const jumpToSection = (group: string) => {
    setCollapsed((c) => (c[group] ? { ...c, [group]: false } : c));
    requestAnimationFrame(() => {
      document
        .getElementById(sectionDomId(group))
        ?.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  };

  return (
    <div className="app">
      <TopBar
        meta={meta}
        device={device}
        rendering={preview.rendering}
        renderMs={preview.renderMs}
        supersample={supersample}
        onSupersample={setSupersample}
        dropping={upload.dropping}
        onDropping={upload.setDropping}
        onFile={upload.onFile}
        randomizeSeedOnOpen={randomizeSeedOnOpen}
        onRandomizeSeedOnOpen={setRandomizeSeedOnOpen}
        onRandomizeSeeds={v.randomizeSeeds}
      />

      <main className="body">
        <Stage
          meta={meta}
          previewUrl={preview.previewUrl}
          sourceUrl={preview.sourceUrl}
          compare={compare}
          onCompare={setCompare}
          split={split}
          onSplit={setSplit}
          showBefore={showBefore}
          onShowBefore={setShowBefore}
          previewFull={preview.previewFull}
          onFile={upload.onFile}
          rendering={preview.rendering}
          job={exporter.job}
          bgLightness={bgLightness}
          canUndo={v.canUndo}
          canRedo={v.canRedo}
          onUndo={v.undo}
          onRedo={v.redo}
          corner={
            <ExportPanel
              meta={meta}
              exportSs={exporter.exportSs}
              onExportSs={exporter.setExportSs}
              format={exporter.format}
              onFormat={exporter.setFormat}
              onExport={exporter.doExport}
              job={exporter.job}
              // The collapse switch rides in the export overlay because that
              // is the overlay it was asked for, and it is the same kind of
              // thing: neither changes a pixel of the render. The section jump
              // menu joins it for the same reason, and sits beside it because
              // the two are the pair of controls that drive the panel rather
              // than the picture.
              headerAside={
                <>
                  <button
                    className="seg"
                    onClick={toggleAllCollapsed}
                    disabled={!grouped.length}
                    title="Collapse or expand every pipeline section in the panel"
                  >
                    {allCollapsed ? "Expand all" : "Collapse all"}
                  </button>
                  <SectionMenu
                    groups={grouped.map((g) => g.group)}
                    onPick={jumpToSection}
                  />
                </>
              }
            />
          }
        />

        <aside className="panel">
          {error && <div className="err">{error}</div>}

          <img
            src={filmGrain16x9}
            alt="Film grain sample"
            className="film-grain-preview"
          />

          {/* Compare and Wipe used to live here. They are on the preview's own
              bar now: they are things you do *to the view*, like zoom, and
              having them in the panel meant looking away from the photo to
              drive a wipe across it. */}
          <ScalePanel
            meta={meta}
            referenceMp={v.referenceMp}
            scaleToRef={scaleToRef}
            scaleOverride={scaleOverride}
            onToggleScaleToRef={() => setScaleToRef((x) => !x)}
            onSetFromPhoto={() => meta && v.setReferenceMp(meta.megapixels)}
            onScaleOverride={(f) => {
              setScaleOverride(f);
              // Setting a factor by hand means wanting it used; leaving it
              // switched off would make the slider do nothing at all.
              if (f !== null) setScaleToRef(true);
            }}
          />

          <Field label="Preview fidelity">
            <button
              className="btn"
              onClick={preview.renderFull}
              disabled={!meta || preview.previewFull || preview.rendering}
              title="Render the whole frame at full resolution"
            >
              {preview.renderingFull
                ? "Rendering 1:1…"
                : preview.previewFull
                  ? "1:1 — up to date"
                  : "Render 1:1"}
            </button>
          </Field>
          <p className="hint">
            Editing renders a{" "}
            {meta ? `${meta.proxy_width}×${meta.proxy_height}` : "proxy"} proxy
            so sliders stay responsive. It predicts structure but cannot resolve
            the finest grain — <strong>Render 1:1</strong> for the exact
            exported pixels, and judge grain there at 100% zoom. Any adjustment
            drops back to the proxy.
          </p>
          <p className="hint">
            Scroll over the photo to zoom about the pointer, drag to pan.
            Neither re-renders.
          </p>

          {/* A view control, sat with the other things that change how the
              preview is *shown* rather than what is in it. The eye takes its
              black point from the whole field of view, so the same photo reads
              contrastier against near-black than against grey -- worth being
              able to move while judging shadow density or halation. */}
          <Field label="Preview background">
            <input
              type="range"
              min={0}
              max={BOARD_LIGHT_MAX}
              step={1}
              value={bgLightness}
              onChange={(e) => setBgLightness(Number(e.target.value))}
              title="Lightness of the chequerboard behind the photo"
            />
            <span className="val">{bgLightness}%</span>
          </Field>

          <Field label="Quality">
            <select
              value={supersample}
              onChange={(e) => setSupersample(Number(e.target.value))}
            >
              <option value={0.5}>0.5× (fastest, soft)</option>
              <option value={1}>1× (fast, aliased grain)</option>
              <option value={1.5}>1.5×</option>
              <option value={2}>2× supersampled</option>
              <option value={3}>3× supersampled (slowest)</option>
            </select>
          </Field>

          <PresetPicker
            schema={schema}
            isOriginal={isNeutral(schema, v.values)}
            onApplyPreset={v.applyPreset}
            onShowOriginal={v.showOriginal}
            onResetAll={v.resetAll}
            onSaveFile={presetFile.savePreset}
            onLoadFile={presetFile.loadPreset}
            fileRef={presetFile.fileRef}
            notice={notice}
          />

          <SliderPanel
            grouped={grouped}
            values={v.values}
            muted={v.muted}
            collapsed={collapsed}
            onToggleCollapsed={(g) =>
              setCollapsed((c) => ({ ...c, [g]: !c[g] }))
            }
            onResetGroup={v.resetGroup}
            onToggleGroup={v.toggleGroup}
            onChange={v.setValue}
            onChangeNow={v.setValueNow}
            onCommit={v.commit}
            luts={luts}
            lut={v.lut}
            onPickLut={lutCtl.pickLut}
            onLoadLutFile={() => lutCtl.fileRef.current?.click()}
          />
          <input
            ref={lutCtl.fileRef}
            type="file"
            accept=".cube"
            hidden
            onChange={(e) => {
              const f = e.target.files?.[0];
              // Cleared for the reason the image and preset pickers are: without
              // it, re-picking the same file fires no change event at all.
              e.target.value = "";
              if (f) lutCtl.onLutFile(f);
            }}
          />
          {/* The export controls used to close the panel here. They are on the
              preview's bottom-right overlay now, for the reason Compare and
              Wipe moved to its top-right one: it is the last thing you do to a
              picture you are looking at, and it was below a scroll. */}
        </aside>
      </main>
    </div>
  );
}
