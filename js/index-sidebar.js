(() => {
  const ensureMarginSidebar = () =>
    document.getElementById("quarto-margin-sidebar");

  const ensureListing = () =>
    window["quarto-listings"] && window["quarto-listings"]["listing-listing"];

  const getCategoryFromHash = () => {
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    return params.get("category") || "";
  };

  const resolveCategoryToken = (token) => {
    if (!token) return "";
    const categoryEls = window.document.querySelectorAll(
      ".quarto-listing-category .category"
    );
    for (const categoryEl of categoryEls) {
      const value = categoryEl.getAttribute("data-category");
      if (value === token) {
        return value;
      }
    }
    const normalizedToken = token.trim().toLowerCase();
    for (const categoryEl of categoryEls) {
      const label = categoryEl.textContent || "";
      if (label.trim().toLowerCase() === normalizedToken) {
        return categoryEl.getAttribute("data-category") || token;
      }
    }
    return token;
  };

  const setCategoryHash = (category) => {
    const base = window.location.pathname + window.location.search;
    const params = new URLSearchParams(window.location.hash.replace(/^#/, ""));
    if (category) {
      params.set("category", category);
    } else {
      params.delete("category");
    }
    const nextHash = params.toString();
    window.history.pushState(null, "", nextHash ? `${base}#${nextHash}` : base);
  };

  const applyCategoryFilter = (category, listing) => {
    const categoryEls = window.document.querySelectorAll(
      ".quarto-listing-category .category"
    );
    for (const categoryEl of categoryEls) {
      const value = categoryEl.getAttribute("data-category");
      categoryEl.classList.toggle("active", value === category);
    }

    if (category === "") {
      listing.filter();
      return;
    }

    listing.filter((item) => {
      const itemValues = item.values();
      if (itemValues.categories !== null) {
        const categories = itemValues.categories.split(",");
        return categories.includes(category);
      }
      return false;
    });
  };

  const wireCategoryHandlers = (listing) => {
    window.quartoListingCategory = (category) => {
      applyCategoryFilter(category, listing);
      setCategoryHash(category);
      return false;
    };

    const categoryEls = window.document.querySelectorAll(
      ".quarto-listing-category .category"
    );
    for (const categoryEl of categoryEls) {
      categoryEl.onclick = () => {
        const category = categoryEl.getAttribute("data-category");
        window.quartoListingCategory(category);
      };
    }

    const categoryTitleEls = window.document.querySelectorAll(
      ".quarto-listing-category-title"
    );
    for (const categoryTitleEl of categoryTitleEls) {
      categoryTitleEl.onclick = () => window.quartoListingCategory("");
    }
  };

  const buildMarginFilter = () => {
    const wrapper = document.createElement("div");
    wrapper.className = "quarto-margin-filter";
    wrapper.innerHTML = `
      <div class="input-group input-group-sm">
        <span class="input-group-text"><i class="bi bi-search"></i></span>
        <input type="text" class="form-control" placeholder="Filter posts">
      </div>
    `;
    return wrapper;
  };

  const placeSeriesUnderCategories = (sidebar) => {
    const series = document.getElementById("series-sidebar-block");
    if (!series) return;

    const categories = sidebar.querySelector(".quarto-listing-category");
    if (categories) {
      categories.insertAdjacentElement("afterend", series);
    } else {
      sidebar.appendChild(series);
    }
    series.style.display = "";
  };

  const placeFilterAboveCategories = (sidebar, listing) => {
    const categoriesTitle = sidebar.querySelector(
      ".quarto-listing-category-title"
    );
    if (!categoriesTitle) return;

    const existing = sidebar.querySelector(".quarto-margin-filter");
    const filter = existing || buildMarginFilter();

    if (!existing) {
      sidebar.insertBefore(filter, categoriesTitle);
    }

    const input = filter.querySelector("input");
    if (!input) return;

    const builtInFilter = window.document.querySelector(
      "#listing-listing .quarto-listing-filter"
    );
    if (builtInFilter) builtInFilter.style.display = "none";

    input.addEventListener("input", () => {
      listing.search(input.value);
    });
  };

  window.document.addEventListener("DOMContentLoaded", () => {
    const sidebar = ensureMarginSidebar();
    if (!sidebar) return;

    const applyHashCategory = (listing) => {
      const categoryToken = getCategoryFromHash();
      if (!categoryToken) return;
      const resolved = resolveCategoryToken(categoryToken);
      applyCategoryFilter(resolved, listing);
    };

    const setup = () => {
      const listing = ensureListing();
      if (!listing) {
        window.setTimeout(setup, 50);
        return;
      }

      placeFilterAboveCategories(sidebar, listing);
      placeSeriesUnderCategories(sidebar);
      wireCategoryHandlers(listing);
      applyHashCategory(listing);

      window.addEventListener("hashchange", () => {
        applyHashCategory(listing);
      });
    };

    setup();
  });
})();
