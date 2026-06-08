// LocalLens Browser UI

const state = {
    selectedPath: "",
    selectedImage: "",
    childrenCache: {},
    expanded: new Set(),
    setupComplete: false,
    currentImages: [],
    pickerExpanded: new Set(),
    pickerChildrenCache: {},
    indexAbort: null,
};

// Main UI elements
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const topkInput = document.getElementById("topk-input");
const folderTree = document.getElementById("folder-tree");
const indexBtn = document.getElementById("index-btn");
const indexStopBtn = document.getElementById("index-stop-btn");
const folderStatus = document.getElementById("folder-status");
const indexProgressWrap = document.getElementById("index-progress-wrap");
const indexProgress = document.getElementById("index-progress");
const indexProgressText = document.getElementById("index-progress-text");
const resultsStatus = document.getElementById("results-status");
const resultsGrid = document.getElementById("results-grid");
const fileList = document.getElementById("file-list");

// Setup elements
const setupBtn = document.getElementById("setup-btn");
const setupBadge = document.getElementById("setup-badge");
const setupModal = document.getElementById("setup-modal");
const setupPath = document.getElementById("setup-path");
const setupActionBtn = document.getElementById("setup-action-btn");
const setupProgress = document.getElementById("setup-progress");
const setupProgressText = document.getElementById("setup-progress-text");
const setupBrowseBtn = document.getElementById("setup-browse-btn");
const setupProcessor = document.getElementById("setup-processor");
const folderPicker = document.getElementById("folder-picker");
const folderPickerTree = document.getElementById("folder-picker-tree");
const folderPickerSelect = document.getElementById("folder-picker-select");
const folderPickerCancel = document.getElementById("folder-picker-cancel");

let pickerSelectedPath = "";

// Event listeners
indexBtn.addEventListener("click", indexFolder);
indexStopBtn.addEventListener("click", () => {
    if (state.indexAbort) state.indexAbort.abort();
});
searchBtn.addEventListener("click", doSearch);
searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
});
setupBtn.addEventListener("click", () => openSetup(false));
setupModal.querySelector(".modal-backdrop").addEventListener("click", closeSetup);
setupActionBtn.addEventListener("click", onSetupAction);
setupPath.addEventListener("input", updateSetupActionButton);
setupProcessor.addEventListener("change", updateSetupActionButton);
setupBrowseBtn.addEventListener("click", openFolderPicker);
folderPickerSelect.addEventListener("click", () => {
    if (pickerSelectedPath) {
        setupPath.value = pickerSelectedPath;
        updateSetupActionButton();
    }
    folderPicker.classList.add("hidden");
});
folderPickerCancel.addEventListener("click", () => folderPicker.classList.add("hidden"));

init();

async function init() {
    await checkSetupStatus();
    loadDrives();
}

// ---- Setup ----

async function checkSetupStatus() {
    try {
        const res = await fetch("/api/setup/status");
        const data = await res.json();
        state.setupComplete = data.complete;
        setupPath.value = data.basePath || data.defaultPath || "";
        populateProcessorOptions(data.detectedProcessor || "cpu", data.processor || "");

        // Snapshot the saved values so the action button can detect edits.
        state.savedBasePath = setupPath.value;
        state.savedProcessor = setupProcessor.value;
        // The processor backend currently loaded in this process. It can
        // only change by restarting the binary.
        state.activeProcessor = data.activeProcessor || "";

        updateSetupBadge();
        updateSetupActionButton();

        if (!data.complete) {
            openSetup(false);
        }
    } catch {}
}

// All processor backends supported by yzma/llama.cpp.
const PROCESSOR_OPTIONS = ["cuda", "vulkan", "metal", "rocm", "cpu"];

function populateProcessorOptions(detected, saved) {
    setupProcessor.innerHTML = "";
    for (const value of PROCESSOR_OPTIONS) {
        const el = document.createElement("option");
        el.value = value;
        el.textContent = value === detected ? `${value} (auto-detected)` : value;
        setupProcessor.appendChild(el);
    }

    // Prefer the saved (config.json) value. Otherwise default to the
    // always-safe "cpu" backend rather than the auto-detected GPU, which the
    // user can still opt into explicitly for faster results.
    setupProcessor.value = PROCESSOR_OPTIONS.includes(saved) ? saved : "cpu";
}

function updateSetupBadge() {
    if (state.setupComplete) {
        setupBadge.textContent = "\u2713";
        setupBadge.className = "ready";
    } else {
        setupBadge.textContent = "\u2717";
        setupBadge.className = "needed";
    }
}

function openSetup() {
    // Don't stomp on a restart-required message that's already in flight.
    if (!state.restartPending) {
        setupProgressText.textContent = state.setupComplete
            ? "Setup complete. You can close this panel."
            : "Setup is required before you can index or search images.";
    }
    setupProgress.classList.remove("hidden");

    updateSetupActionButton();
    setupModal.classList.remove("hidden");
}

function closeSetup() {
    setupModal.classList.add("hidden");
}

// hasPendingChanges reports whether the form values differ from what's
// currently saved in config.json. Used to force a re-run of setup so the
// new basePath/processor actually take effect.
function hasPendingChanges() {
    return (
        setupPath.value.trim() !== (state.savedBasePath || "") ||
        setupProcessor.value !== (state.savedProcessor || "")
    );
}

// processorSwitchRequiresRestart returns true when the user picked a
// processor different from the one currently saved in config.json. Any
// change to the persisted processor requires a restart because the Kronk
// SDK can't swap llama.cpp libraries mid-process.
function processorSwitchRequiresRestart() {
    const saved = state.savedProcessor || "";
    return setupProcessor.value && setupProcessor.value !== saved;
}

// updateSetupActionButton flips the single action button between
// "Close" (no work needed), the setup-running label (work needed), a
// restart-required label (processor switch requested), and "Quit" once
// the server has confirmed the restart is required.
function updateSetupActionButton() {
    if (state.restartPending) {
        setupActionBtn.textContent = "Quit LocalLens";
        setupActionBtn.dataset.action = "quit";
        return;
    }

    const pending = hasPendingChanges();
    const needsRun = !state.setupComplete || pending;

    if (!needsRun) {
        setupActionBtn.textContent = "Close";
        setupActionBtn.dataset.action = "close";
        return;
    }

    if (state.setupComplete && processorSwitchRequiresRestart()) {
        setupActionBtn.textContent = "Install & Save (restart required)";
        setupActionBtn.dataset.action = "run";
        return;
    }

    setupActionBtn.textContent = state.setupComplete
        ? "Apply & Re-run Setup"
        : "Download & Install";
    setupActionBtn.dataset.action = "run";
}

function onSetupAction() {
    switch (setupActionBtn.dataset.action) {
        case "close":
            closeSetup();
            return;
        case "quit":
            quitApp();
            return;
        default:
            runSetup();
    }
}

async function quitApp() {
    setupActionBtn.disabled = true;
    setupProgressText.textContent = "Quitting...";
    try {
        await fetch("/api/quit", { method: "POST" });
    } catch {
        // Server closed the connection — that's the expected outcome.
    }
    // The browser tab is left with a dead backend; close it if the
    // window was opened programmatically, otherwise show a hint.
    window.close();
    setupProgressText.textContent = "LocalLens has stopped. You can close this tab and relaunch the app.";
}

async function runSetup() {
    const basePath = setupPath.value.trim();
    if (!basePath) {
        setupProgressText.textContent = "Please enter a download location.";
        setupProgress.classList.remove("hidden");
        return;
    }

    setupActionBtn.disabled = true;
    setupProgress.classList.remove("hidden");
    setupProgressText.textContent = "Starting download...";

    try {
        const res = await fetch("/api/setup/run", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ basePath, processor: setupProcessor.value }),
        });

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                try {
                    const event = JSON.parse(line.slice(6));
                    handleSetupEvent(event);
                } catch {}
            }
        }

        // Re-pull status so the saved-values snapshot reflects what was
        // just persisted; the button flips back to "Close" automatically.
        await checkSetupStatus();
    } catch {
        setupProgressText.textContent = "Setup failed. Check your connection and try again.";
    } finally {
        setupActionBtn.disabled = false;
        updateSetupActionButton();
    }
}

function handleSetupEvent(event) {
    const labels = {
        libs: "AI runtime libraries",
        models: "AI models",
        init: "Initializing service",
        done: "Setup complete",
    };

    const label = labels[event.step] || event.step;

    // The server emits this when the requested processor differs from the
    // one already loaded in the running process. New libs + config are
    // persisted, but the swap only takes effect after a relaunch.
    if (event.step === "restart_required") {
        setupProgressText.textContent = event.status;
        state.setupComplete = true;
        state.restartPending = true;
        updateSetupBadge();
        updateSetupActionButton();
        return;
    }

    if (event.status === "complete" && event.step === "done") {
        setupProgressText.textContent = "Setup complete. You can close this panel.";
        state.setupComplete = true;
        updateSetupBadge();
        return;
    }

    if (event.status.startsWith("error:")) {
        setupProgressText.textContent = `Error during ${label}: ${event.status.slice(7)}`;
        return;
    }

    if (event.status === "downloading") {
        setupProgressText.textContent = `Downloading ${label}...`;
    } else if (event.status === "complete") {
        setupProgressText.textContent = `${label} — done.`;
    } else if (event.status === "initializing") {
        setupProgressText.textContent = `${label}...`;
    }
}

// ---- Folder Picker (tree-based, like main tree) ----

async function openFolderPicker() {
    pickerSelectedPath = setupPath.value.trim();
    state.pickerExpanded = new Set();
    state.pickerChildrenCache = {};
    folderPicker.classList.remove("hidden");
    folderPickerTree.innerHTML = "";

    const data = await fetchBrowse("");
    if (!data) return;

    for (const folder of (data.folders || [])) {
        const node = createPickerTreeNode(folder.name, folder.path, 0);
        folderPickerTree.appendChild(node);
    }
}

function createPickerTreeNode(name, path, depth) {
    const wrapper = document.createElement("div");
    wrapper.className = "tree-node";

    const row = document.createElement("div");
    row.className = "tree-row";
    if (path === pickerSelectedPath) row.classList.add("selected");
    row.style.paddingLeft = (8 + depth * 16) + "px";

    const arrow = document.createElement("span");
    arrow.className = "tree-arrow";
    arrow.textContent = "\u25B6";
    if (state.pickerExpanded.has(path)) arrow.classList.add("expanded");
    arrow.addEventListener("click", (e) => {
        e.stopPropagation();
        togglePickerExpand(path, wrapper, depth);
    });

    const icon = document.createElement("span");
    icon.className = "tree-icon";
    icon.textContent = "\uD83D\uDCC1";

    const label = document.createElement("span");
    label.className = "tree-label";
    label.textContent = name;
    label.title = path;

    row.appendChild(arrow);
    row.appendChild(icon);
    row.appendChild(label);

    row.addEventListener("click", () => {
        pickerSelectedPath = path;
        setupPath.value = path;
        folderPickerTree.querySelectorAll(".tree-row.selected").forEach(el => el.classList.remove("selected"));
        row.classList.add("selected");
    });

    wrapper.appendChild(row);

    if (state.pickerExpanded.has(path)) {
        renderPickerChildNodes(wrapper, path, depth);
    }

    return wrapper;
}

async function togglePickerExpand(path, wrapper, depth) {
    if (state.pickerExpanded.has(path)) {
        state.pickerExpanded.delete(path);
        const children = wrapper.querySelector(".tree-children");
        if (children) children.remove();
        const arrow = wrapper.querySelector(".tree-arrow");
        if (arrow) arrow.classList.remove("expanded");
    } else {
        state.pickerExpanded.add(path);
        const arrow = wrapper.querySelector(".tree-arrow");
        if (arrow) arrow.classList.add("expanded");
        await renderPickerChildNodes(wrapper, path, depth);
        scrollExpandedIntoView(wrapper);
    }
}

// scrollExpandedIntoView scrolls the enclosing scrollable container so the
// last child of a freshly-expanded node becomes visible. Uses block:nearest
// so already-visible content stays put and only off-screen children cause
// the panel to scroll down.
function scrollExpandedIntoView(wrapper) {
    const children = wrapper.querySelector(".tree-children");
    if (!children) return;
    const last = children.lastElementChild;
    if (!last) return;
    last.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

async function renderPickerChildNodes(wrapper, path, depth) {
    let children = wrapper.querySelector(".tree-children");
    if (children) return;

    let data = state.pickerChildrenCache[path];
    if (!data) {
        data = await fetchBrowse(path);
        if (!data) return;
        state.pickerChildrenCache[path] = data;
    }

    children = document.createElement("div");
    children.className = "tree-children";

    for (const folder of (data.folders || [])) {
        const node = createPickerTreeNode(folder.name, folder.path, depth + 1);
        children.appendChild(node);
    }

    wrapper.appendChild(children);
}

function requireSetup() {
    if (!state.setupComplete) {
        openSetup(true);
        return true;
    }
    return false;
}

// ---- Tree Navigation ----

async function loadDrives() {
    const data = await fetchBrowse("");
    if (!data) return;

    folderTree.innerHTML = "";
    for (const folder of (data.folders || [])) {
        const node = createTreeNode(folder.name, folder.path, 0);
        folderTree.appendChild(node);
    }
}

async function fetchBrowse(path) {
    try {
        const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
        return await res.json();
    } catch {
        return null;
    }
}

async function loadChildren(path) {
    if (state.childrenCache[path]) return state.childrenCache[path];

    const data = await fetchBrowse(path);
    if (!data) return { folders: [], images: [] };

    state.childrenCache[path] = data;
    return data;
}

function createTreeNode(name, path, depth) {
    const wrapper = document.createElement("div");
    wrapper.className = "tree-node";

    const row = document.createElement("div");
    row.className = "tree-row";
    if (path === state.selectedPath) row.classList.add("selected");
    row.style.paddingLeft = (8 + depth * 16) + "px";

    const arrow = document.createElement("span");
    arrow.className = "tree-arrow";
    arrow.textContent = "\u25B6";
    if (state.expanded.has(path)) arrow.classList.add("expanded");
    arrow.addEventListener("click", (e) => {
        e.stopPropagation();
        toggleExpand(path, wrapper, depth);
    });

    const icon = document.createElement("span");
    icon.className = "tree-icon";
    icon.textContent = "\uD83D\uDCC1";

    const label = document.createElement("span");
    label.className = "tree-label";
    label.textContent = name;
    label.title = path;

    row.appendChild(arrow);
    row.appendChild(icon);
    row.appendChild(label);

    row.addEventListener("click", () => selectFolder(path));

    wrapper.appendChild(row);

    if (state.expanded.has(path)) {
        renderChildNodes(wrapper, path, depth);
    }

    return wrapper;
}

async function toggleExpand(path, wrapper, depth) {
    if (state.expanded.has(path)) {
        state.expanded.delete(path);
        const children = wrapper.querySelector(".tree-children");
        if (children) children.remove();
        const arrow = wrapper.querySelector(".tree-arrow");
        if (arrow) arrow.classList.remove("expanded");
    } else {
        state.expanded.add(path);
        const arrow = wrapper.querySelector(".tree-arrow");
        if (arrow) arrow.classList.add("expanded");
        await renderChildNodes(wrapper, path, depth);
        scrollExpandedIntoView(wrapper);
    }
}

async function renderChildNodes(wrapper, path, depth) {
    let children = wrapper.querySelector(".tree-children");
    if (children) return;

    const data = await loadChildren(path);

    children = document.createElement("div");
    children.className = "tree-children";

    for (const folder of (data.folders || [])) {
        const node = createTreeNode(folder.name, folder.path, depth + 1);
        children.appendChild(node);
    }

    wrapper.appendChild(children);
}

async function selectFolder(path) {
    state.selectedPath = path;
    state.selectedImage = "";

    document.querySelectorAll("#folder-tree .tree-row.selected").forEach((el) => el.classList.remove("selected"));
    document.querySelectorAll("#folder-tree .tree-row").forEach((el) => {
        const label = el.querySelector(".tree-label");
        if (label && label.title === path) el.classList.add("selected");
    });

    if (!state.expanded.has(path)) {
        const wrapper = findWrapperForPath(path);
        if (wrapper) {
            const depth = getDepth(wrapper);
            await toggleExpand(path, wrapper, depth);
        }
    }

    searchInput.disabled = false;
    searchBtn.disabled = false;
    indexBtn.disabled = false;

    loadIndexInfo(path);
    loadFolderImages(path);
}

function findWrapperForPath(path) {
    const labels = document.querySelectorAll("#folder-tree .tree-label");
    for (const label of labels) {
        if (label.title === path) return label.closest(".tree-node");
    }
    return null;
}

function getDepth(wrapper) {
    let depth = 0;
    let el = wrapper.parentElement;
    while (el) {
        if (el.classList && el.classList.contains("tree-children")) depth++;
        el = el.parentElement;
    }
    return depth;
}

async function loadFolderImages(path) {
    const data = await loadChildren(path);
    const images = data.images || [];
    state.currentImages = images;

    if (images.length > 0) {
        resultsStatus.textContent = `${images.length} images in folder`;
    } else {
        resultsStatus.textContent = "No images in this folder";
    }

    renderFileList(images);
    renderImages(images);
}

// ---- File List Panel ----

function renderFileList(images) {
    fileList.innerHTML = "";

    for (let i = 0; i < images.length; i++) {
        const img = images[i];
        const row = document.createElement("div");
        row.className = "file-row";
        row.dataset.path = img.path;
        if (img.path === state.selectedImage) row.classList.add("selected");

        const icon = document.createElement("span");
        icon.className = "file-icon";
        icon.textContent = "\uD83D\uDDBC";

        const name = document.createElement("span");
        name.className = "file-name";
        name.textContent = img.name;
        name.title = img.path;

        const status = document.createElement("span");
        status.className = "file-status";
        status.textContent = img.indexed ? "\u2705" : "\u2B1C";

        row.appendChild(icon);
        row.appendChild(name);
        row.appendChild(status);

        row.addEventListener("click", () => selectImage(img.path));
        row.addEventListener("dblclick", () => openInExplorer(img.path));

        fileList.appendChild(row);
    }
}

function renderSearchFileList(results) {
    fileList.innerHTML = "";

    for (let i = 0; i < results.length; i++) {
        const r = results[i];
        const name = fileName(r.Path);
        const row = document.createElement("div");
        row.className = "file-row";
        row.dataset.path = r.Path;
        if (r.Path === state.selectedImage) row.classList.add("selected");

        const icon = document.createElement("span");
        icon.className = "file-icon";
        icon.textContent = "\uD83D\uDDBC";

        const nameEl = document.createElement("span");
        nameEl.className = "file-name";
        nameEl.textContent = name;
        nameEl.title = r.Path;

        const score = document.createElement("span");
        score.className = "file-status";
        score.textContent = `${(r.Score * 100).toFixed(0)}%`;
        score.style.color = "#00d9ff";

        row.appendChild(icon);
        row.appendChild(nameEl);
        row.appendChild(score);

        row.addEventListener("click", () => selectImage(r.Path));
        row.addEventListener("dblclick", () => openInExplorer(r.Path));

        fileList.appendChild(row);
    }
}

// ---- Selection Sync ----

function selectImage(path) {
    state.selectedImage = path;

    // Highlight in file list + auto-scroll
    fileList.querySelectorAll(".file-row.selected").forEach(el => el.classList.remove("selected"));
    const fileRow = fileList.querySelector(`.file-row[data-path="${CSS.escape(path)}"]`);
    if (fileRow) {
        fileRow.classList.add("selected");
        fileRow.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }

    // Highlight in image grid + auto-scroll
    resultsGrid.querySelectorAll(".result-card.selected").forEach(el => el.classList.remove("selected"));
    const card = resultsGrid.querySelector(`.result-card[data-path="${CSS.escape(path)}"]`);
    if (card) {
        card.classList.add("selected");
        card.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
}

// ---- Controls ----

async function loadIndexInfo(folder) {
    try {
        const res = await fetch(`/api/index-info?folder=${encodeURIComponent(folder)}`);
        if (res.status === 503) {
            folderStatus.textContent = "Setup required";
            return;
        }
        const data = await res.json();
        folderStatus.textContent = data.count > 0
            ? `${data.count} images indexed`
            : "Not indexed yet";
    } catch {
        folderStatus.textContent = "";
    }
}

// ---- Images ----

function renderImages(images) {
    resultsGrid.innerHTML = "";

    for (const img of images) {
        const card = createImageCard(img.path, img.name, img.indexed, null);
        resultsGrid.appendChild(card);
    }
}

function renderSearchResults(results) {
    resultsGrid.innerHTML = "";

    for (const r of results) {
        const name = fileName(r.Path);
        const card = createImageCard(r.Path, name, true, r.Score);
        resultsGrid.appendChild(card);
    }
}

function createImageCard(path, name, indexed, score) {
    const card = document.createElement("div");
    card.className = "result-card";
    card.dataset.path = path;
    if (path === state.selectedImage) card.classList.add("selected");

    card.addEventListener("click", () => selectImage(path));
    card.addEventListener("dblclick", () => openInExplorer(path));
    card.title = name;

    const img = document.createElement("img");
    img.src = `/api/images?path=${encodeURIComponent(path)}`;
    img.alt = name;
    img.loading = "lazy";

    const info = document.createElement("div");
    info.className = "info";

    const statusIcon = document.createElement("span");
    statusIcon.className = "status-icon";
    statusIcon.textContent = indexed ? "\u2705" : "\u2B1C";
    statusIcon.title = indexed ? "Indexed" : "Not indexed";

    const nameEl = document.createElement("span");
    nameEl.className = "name";
    nameEl.textContent = name;
    nameEl.title = name;

    info.appendChild(statusIcon);
    info.appendChild(nameEl);

    if (score !== null) {
        const scoreEl = document.createElement("span");
        scoreEl.className = "score";
        scoreEl.textContent = `${(score * 100).toFixed(1)}%`;
        info.appendChild(scoreEl);
    }

    card.appendChild(img);
    card.appendChild(info);
    return card;
}

function fileName(path) {
    const parts = path.replace(/\\/g, "/").split("/");
    return parts[parts.length - 1];
}

async function openInExplorer(path) {
    try {
        await fetch("/api/open", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ path }),
        });
    } catch {}
}

// ---- Index ----

async function indexFolder() {
    if (requireSetup()) return;
    if (!state.selectedPath) return;

    indexBtn.disabled = true;
    indexStopBtn.hidden = false;
    folderStatus.textContent = "Starting...";
    showProgress(0, 0, 0);

    const controller = new AbortController();
    state.indexAbort = controller;

    let finalCount = 0;
    let stopped = false;

    try {
        const res = await fetch("/api/index", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder: state.selectedPath }),
            signal: controller.signal,
        });

        if (res.status === 503) {
            openSetup(true);
            return;
        }

        if (!res.ok) {
            folderStatus.textContent = "Indexing failed";
            return;
        }

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        outer: while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split("\n");
            buffer = lines.pop();

            for (const line of lines) {
                if (!line.startsWith("data: ")) continue;
                let event;
                try {
                    event = JSON.parse(line.slice(6));
                } catch {
                    continue;
                }

                switch (event.type) {
                    case "started":
                        folderStatus.textContent = "Loading vision model...";
                        break;
                    case "progress":
                        if (event.stage === "describing") {
                            // Slow operation — show activity in the status.
                            // Progress bar/ETA are not touched here because
                            // the describing event carries no timing data,
                            // so we'd just overwrite the previous valid ETA
                            // with "—".
                            folderStatus.textContent = formatDescribing(event);
                        } else if (event.stage === "indexed") {
                            // stage === "indexed": image fully saved.
                            // Update progress bar (value + ETA) and flip
                            // the per-image checkbox.
                            finalCount = event.done;
                            markImageIndexed(event.current);
                            showProgress(event.processed ?? event.done, event.total, event.etaMs);
                        } else if (event.stage === "failed") {
                            finalCount = event.done;
                            folderStatus.textContent = `${event.done} indexed, ${event.failed} failed`;
                            showProgress(event.processed ?? (event.done + event.failed), event.total, event.etaMs);
                        }
                        break;
                    case "done":
                        finalCount = event.count;
                        folderStatus.textContent = formatIndexSummary(event);
                        hideProgress();
                        break outer;
                    case "cancelled":
                        finalCount = event.count;
                        stopped = true;
                        folderStatus.textContent = `Stopped — ${formatIndexSummary(event)}`;
                        hideProgress();
                        break outer;
                    case "error":
                        finalCount = event.count || 0;
                        folderStatus.textContent = `Indexing failed: ${event.error}`;
                        hideProgress();
                        break outer;
                }
            }
        }
    } catch (err) {
        if (err.name === "AbortError") {
            stopped = true;
            folderStatus.textContent = `Stopped — ${finalCount} images indexed`;
        } else {
            folderStatus.textContent = "Indexing failed";
        }
        hideProgress();
    } finally {
        state.indexAbort = null;
        indexBtn.disabled = false;
        indexStopBtn.hidden = true;
        // Refresh the file list so newly-indexed images show up.
        delete state.childrenCache[state.selectedPath];
        if (state.selectedPath) loadFolderImages(state.selectedPath);
        // Note: stopped is logged in folderStatus already; no extra UI work.
        void stopped;
    }
}

function showProgress(done, total, etaMs) {
    indexProgressWrap.hidden = false;
    indexProgress.value = total > 0 ? Math.floor((done / total) * 100) : 0;
    const eta = etaMs > 0 ? formatETA(etaMs) : "—";
    indexProgressText.textContent = `${done}/${total} \u2022 ETA ${eta}`;
}

function hideProgress() {
    indexProgressWrap.hidden = true;
    indexProgress.value = 0;
    indexProgressText.textContent = "";
}

function formatDescribing(p) {
    const next = p.done + 1;
    const file = p.current ? ` — ${shortenPath(p.current)}` : "";
    return `Describing image ${next}/${p.total}${file}`;
}

function formatIndexSummary(event) {
    const count = event.count ?? event.indexedTotal ?? 0;
    const added = event.added ?? 0;
    const failed = event.failed ?? 0;
    if (failed > 0) {
        return `${count} images indexed (${added} new, ${failed} failed)`;
    }
    return `${count} images indexed`;
}

// markImageIndexed flips the ✅ marker on the row and the result card for the
// just-indexed image, plus updates the cached image record so subsequent
// renders preserve the state. Called per image as "indexed" events arrive.
function markImageIndexed(path) {
    if (!path) return;

    for (const img of state.currentImages) {
        if (img.path === path) {
            img.indexed = true;
            break;
        }
    }

    const row = fileList.querySelector(`.file-row[data-path="${cssEscape(path)}"]`);
    if (row) {
        const status = row.querySelector(".file-status");
        if (status) status.textContent = "\u2705";
    }

    const card = resultsGrid.querySelector(`.result-card[data-path="${cssEscape(path)}"]`);
    if (card) {
        const icon = card.querySelector(".status-icon");
        if (icon) {
            icon.textContent = "\u2705";
            icon.title = "Indexed";
        }
    }
}

// cssEscape quotes a string for use inside an attribute selector. Falls back
// to a manual escape when CSS.escape isn't available (very old browsers).
function cssEscape(s) {
    if (window.CSS && CSS.escape) return CSS.escape(s);
    return s.replace(/(["\\])/g, "\\$1");
}

function formatETA(ms) {
    if (!ms || ms <= 0) return "—";
    const sec = Math.round(ms / 1000);
    if (sec < 60) return `${sec}s`;
    const min = Math.floor(sec / 60);
    const rem = sec % 60;
    if (min < 60) return rem ? `${min}m ${rem}s` : `${min}m`;
    const hr = Math.floor(min / 60);
    const minRem = min % 60;
    return minRem ? `${hr}h ${minRem}m` : `${hr}h`;
}

function shortenPath(p) {
    const i = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
    return i >= 0 ? p.slice(i + 1) : p;
}

// ---- Search ----

async function doSearch() {
    if (requireSetup()) return;

    const query = searchInput.value.trim();
    if (!query || !state.selectedPath) return;

    const k = parseInt(topkInput.value) || 10;

    searchBtn.disabled = true;
    resultsStatus.textContent = "Searching...";
    resultsGrid.innerHTML = "";
    fileList.innerHTML = "";

    try {
        const res = await fetch(
            `/api/search?q=${encodeURIComponent(query)}&folder=${encodeURIComponent(state.selectedPath)}&k=${k}`
        );

        if (res.status === 503) {
            openSetup(true);
            return;
        }

        const results = await res.json();

        if (!results || results.length === 0) {
            resultsStatus.textContent = "No results found";
            return;
        }

        resultsStatus.textContent = `${results.length} results for "${query}"`;
        state.selectedImage = "";
        renderSearchFileList(results);
        renderSearchResults(results);
    } catch {
        resultsStatus.textContent = "Search failed";
    } finally {
        searchBtn.disabled = false;
    }
}
