// LocalLens Browser UI

const state = {
    selectedPath: "",
    expanded: new Set(),
    childrenCache: {},
};

const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const topkInput = document.getElementById("topk-input");
const folderTree = document.getElementById("folder-tree");
const indexBtn = document.getElementById("index-btn");
const folderStatus = document.getElementById("folder-status");
const resultsStatus = document.getElementById("results-status");
const resultsGrid = document.getElementById("results-grid");

indexBtn.addEventListener("click", indexFolder);
searchBtn.addEventListener("click", doSearch);
searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
});

loadDrives();

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

    document.querySelectorAll(".tree-row.selected").forEach((el) => el.classList.remove("selected"));
    document.querySelectorAll(".tree-row").forEach((el) => {
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
    const labels = document.querySelectorAll(".tree-label");
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

    if (images.length > 0) {
        resultsStatus.textContent = `${images.length} images in folder`;
    } else {
        resultsStatus.textContent = "No images in this folder";
    }

    renderImages(images);
}

// ---- Controls ----

async function loadIndexInfo(folder) {
    try {
        const res = await fetch(`/api/index-info?folder=${encodeURIComponent(folder)}`);
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
    card.addEventListener("dblclick", () => openInExplorer(path));
    card.title = "Double-click to open in Explorer";

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
    if (!state.selectedPath) return;

    indexBtn.disabled = true;
    folderStatus.textContent = "Indexing... this may take a while";

    try {
        const res = await fetch("/api/index", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder: state.selectedPath }),
        });
        const data = await res.json();

        if (res.ok) {
            folderStatus.textContent = `${data.count} images indexed`;
            delete state.childrenCache[state.selectedPath];
            loadFolderImages(state.selectedPath);
        } else {
            folderStatus.textContent = "Indexing failed";
        }
    } catch {
        folderStatus.textContent = "Indexing failed";
    } finally {
        indexBtn.disabled = false;
    }
}

// ---- Search ----

async function doSearch() {
    const query = searchInput.value.trim();
    if (!query || !state.selectedPath) return;

    const k = parseInt(topkInput.value) || 10;

    searchBtn.disabled = true;
    resultsStatus.textContent = "Searching...";
    resultsGrid.innerHTML = "";

    try {
        const res = await fetch(
            `/api/search?q=${encodeURIComponent(query)}&folder=${encodeURIComponent(state.selectedPath)}&k=${k}`
        );
        const results = await res.json();

        if (!results || results.length === 0) {
            resultsStatus.textContent = "No results found";
            return;
        }

        resultsStatus.textContent = `${results.length} results for "${query}"`;
        renderSearchResults(results);
    } catch {
        resultsStatus.textContent = "Search failed";
    } finally {
        searchBtn.disabled = false;
    }
}
