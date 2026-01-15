(() => {
  const ensureMarginSidebar = () =>
    document.getElementById("quarto-margin-sidebar");

  const ensureListing = () =>
    window["quarto-listings"] && window["quarto-listings"]["listing-listing"];

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

    const setup = () => {
      const listing = ensureListing();
      if (!listing) {
        window.setTimeout(setup, 50);
        return;
      }

      placeFilterAboveCategories(sidebar, listing);
      placeSeriesUnderCategories(sidebar);
    };

    setup();
  });
})();
