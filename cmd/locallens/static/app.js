// LocalLens Browser UI

const state = {
    currentPath: "",
    images: [],
};

// DOM elements
const searchInput = document.getElementById("search-input");
const searchBtn = document.getElementById("search-btn");
const folderTree = document.getElementById("folder-tree");
const indexBtn = document.getElementById("index-btn");
const folderStatus = document.getElementById("folder-status");
const resultsStatus = document.getElementById("results-status");
const resultsGrid = document.getElementById("results-grid");

// Events
indexBtn.addEventListener("click", indexFolder);
searchBtn.addEventListener("click", doSearch);
searchInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") doSearch();
});

// Start by listing drives.
navigateTo("");

async function navigateTo(path) {
    state.currentPath = path;

    try {
        const res = await fetch(`/api/browse?path=${encodeURIComponent(path)}`);
        const data = await res.json();
        renderFolderPanel(data);
        renderImages(data.images || []);
        updateControls(data);
    } catch {
        folderStatus.textContent = "Failed to browse folder";
    }
}

function renderFolderPanel(data) {
    folderTree.innerHTML = "";

    if (data.parent !== undefined && data.parent !== "") {
        const up = document.createElement("div");
        up.className = "folder-item folder-up";
        up.textContent = "\u2190 ..";
        up.addEventListener("click", () => navigateTo(data.parent));
        folderTree.appendChild(up);
    } else if (data.current !== "") {
        const root = document.createElement("div");
        root.className = "folder-item folder-up";
        root.textContent = "\u2190 Drives";
        root.addEventListener("click", () => navigateTo(""));
        folderTree.appendChild(root);
    }

    if (data.current) {
        const current = document.createElement("div");
        current.className = "folder-current";
        current.textContent = data.current;
        current.title = data.current;
        folderTree.appendChild(current);
    }

    for (const folder of (data.folders || [])) {
        const div = document.createElement("div");
        div.className = "folder-item";
        div.textContent = "\uD83D\uDCC1 " + folder.name;
        div.title = folder.path;
        div.addEventListener("click", () => navigateTo(folder.path));
        folderTree.appendChild(div);
    }
}

function updateControls(data) {
    const hasPath = data.current !== "";
    searchInput.disabled = !hasPath;
    searchBtn.disabled = !hasPath;
    indexBtn.disabled = !hasPath;

    if (hasPath) {
        loadIndexInfo(data.current);
    } else {
        folderStatus.textContent = "";
    }

    const imageCount = (data.images || []).length;
    if (hasPath && imageCount > 0) {
        resultsStatus.textContent = `${imageCount} images in folder`;
    } else if (hasPath) {
        resultsStatus.textContent = "No images in this folder";
    } else {
        resultsStatus.textContent = "Select a folder to browse images";
    }
}

function renderImages(images) {
    resultsGrid.innerHTML = "";
    state.images = images;

    for (const imgPath of images) {
        const card = document.createElement("div");
        card.className = "result-card";

        const img = document.createElement("img");
        img.src = `/api/images?path=${encodeURIComponent(imgPath)}`;
        img.alt = fileName(imgPath);
        img.loading = "lazy";

        const info = document.createElement("div");
        info.className = "info";

        const name = document.createElement("div");
        name.className = "description";
        name.textContent = fileName(imgPath);

        info.appendChild(name);
        card.appendChild(img);
        card.appendChild(info);
        resultsGrid.appendChild(card);
    }
}

function renderSearchResults(results) {
    resultsGrid.innerHTML = "";

    for (const r of results) {
        const card = document.createElement("div");
        card.className = "result-card";

        const img = document.createElement("img");
        img.src = `/api/images?path=${encodeURIComponent(r.Path)}`;
        img.alt = r.Description;
        img.loading = "lazy";

        const info = document.createElement("div");
        info.className = "info";

        const desc = document.createElement("div");
        desc.className = "description";
        desc.textContent = r.Description;

        const score = document.createElement("div");
        score.className = "score";
        score.textContent = `Score: ${(r.Score * 100).toFixed(1)}%`;

        info.appendChild(desc);
        info.appendChild(score);
        card.appendChild(img);
        card.appendChild(info);
        resultsGrid.appendChild(card);
    }
}

function fileName(path) {
    const parts = path.replace(/\\/g, "/").split("/");
    return parts[parts.length - 1];
}

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

async function indexFolder() {
    if (!state.currentPath) return;

    indexBtn.disabled = true;
    folderStatus.textContent = "Indexing... this may take a while";

    try {
        const res = await fetch("/api/index", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ folder: state.currentPath }),
        });
        const data = await res.json();

        if (res.ok) {
            folderStatus.textContent = `${data.count} images indexed`;
        } else {
            folderStatus.textContent = "Indexing failed";
        }
    } catch {
        folderStatus.textContent = "Indexing failed";
    } finally {
        indexBtn.disabled = false;
    }
}

async function doSearch() {
    const query = searchInput.value.trim();
    if (!query || !state.currentPath) return;

    searchBtn.disabled = true;
    resultsStatus.textContent = "Searching...";
    resultsGrid.innerHTML = "";

    try {
        const res = await fetch(
            `/api/search?q=${encodeURIComponent(query)}&folder=${encodeURIComponent(state.currentPath)}`
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
