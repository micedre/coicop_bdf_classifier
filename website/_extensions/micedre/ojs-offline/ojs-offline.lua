-- ojs-offline.lua
-- Quarto filter to inject offline Observable JS dependencies
-- This filter intercepts CDN requests BEFORE quarto-ojs-runtime.js loads

-- Always inject for HTML output - OJS blocks are processed before this filter
-- runs, so we can't reliably detect them. The interceptor is harmless if unused.

-- Inject dependencies at the Meta stage
function Meta(meta)
  -- Only inject for HTML output
  if not quarto.doc.is_format("html") then
    return meta
  end

  -- Get the path to the extension's resources
  -- quarto.project.offset can be "." or empty, handle both cases
  local offset = quarto.project.offset or ""
  if offset == "." then
    offset = ""
  elseif offset ~= "" and not offset:match("/$") then
    offset = offset .. "/"
  end
  local extension_dir = offset .. "_extensions/ojs-offline/"

  -- Read the dependency map
  local map_path = extension_dir .. "resources/dependency-map.json"
  local map_file = io.open(map_path, "r")
  local dep_map_json = "{}"
  if map_file then
    dep_map_json = map_file:read("*a")
    map_file:close()
  else
    quarto.log.warning("OJS Offline: Could not read dependency-map.json from " .. map_path)
  end

  -- The libs will be at site_libs/quarto-contrib/ojs-offline-libs-1.0.0/libs/
  -- We need to update the dependency map paths to match this structure
  local libs_base = "site_libs/quarto-contrib/ojs-offline-libs-1.0.0/libs/"

  -- Transform the dependency map paths
  dep_map_json = dep_map_json:gsub('"ojs%-offline%-libs/', '"' .. libs_base)

  -- Create the inline script that intercepts fetch calls
  -- This MUST run before quarto-ojs-runtime.js to intercept CDN requests
  -- Using [=[ ]=] delimiters to avoid conflicts with JavaScript brackets
  local script_part1 = [=[
<script>
(function() {
  'use strict';

  // Dependency map loaded from extension
  const depMap = ]=]

  local script_part2 = [=[;

  // CDN URLs to intercept
  const CDN_JSDELIVR = 'https://cdn.jsdelivr.net/npm/';
  const CDN_OBSERVABLE = 'https://cdn.observableusercontent.com/npm/';

  // Version aliases - map major versions to specific versions we have
  const versionAliases = {
    'd3@7': 'd3@7.8.5',
    'd3@6': 'd3@7.8.5',
    'lodash@4': 'lodash@4.17.21',
    'htl@0': 'htl@0.3.1',
    '@observablehq/inputs@0': '@observablehq/inputs@0.10.6',
    '@observablehq/plot@0': '@observablehq/plot@0.6.11',
    'vega@5': 'vega@5.22.1',
    'vega-lite@5': 'vega-lite@5.6.0'
  };

  // Find local path for a package specifier
  function findLocalPath(specifier) {
    // Direct lookup
    if (depMap[specifier]) {
      return depMap[specifier];
    }

    // Try version alias
    const atIndex = specifier.lastIndexOf('@');
    if (atIndex > 0) {
      const name = specifier.substring(0, atIndex);
      const version = specifier.substring(atIndex + 1);
      // Check for major version alias (e.g., d3@7 -> d3@7.8.5)
      const majorVersion = version.split('.')[0];
      const aliasKey = name + '@' + majorVersion;
      if (versionAliases[aliasKey]) {
        const aliased = versionAliases[aliasKey];
        if (depMap[aliased]) {
          return depMap[aliased];
        }
      }
    }

    // Try without version (bare package name)
    const bareMatch = specifier.match(/^(@?[^@\/]+(?:\/[^@\/]+)?)/);
    if (bareMatch && depMap[bareMatch[1]]) {
      return depMap[bareMatch[1]];
    }

    // Try to find any version of the package
    const pkgName = specifier.split('@')[0] || specifier.split('/')[0];
    for (const key in depMap) {
      if (key.startsWith(pkgName + '@') || key === pkgName) {
        return depMap[key];
      }
    }

    return null;
  }

  // Convert CDN URL to local path
  function cdnToLocal(url) {
    let pkg = null;
    if (url.startsWith(CDN_JSDELIVR)) {
      pkg = url.slice(CDN_JSDELIVR.length);
    } else if (url.startsWith(CDN_OBSERVABLE)) {
      pkg = url.slice(CDN_OBSERVABLE.length);
    }

    if (pkg) {
      // Handle package.json requests
      if (pkg.endsWith('/package.json')) {
        // Return a fake package.json response - we'll handle this in fetch override
        return { isPackageJson: true, pkg: pkg.replace('/package.json', '') };
      }

      const localPath = findLocalPath(pkg);
      if (localPath) {
        return localPath;
      }
    }
    return null;
  }

  // Override fetch to intercept CDN requests
  const originalFetch = window.fetch;
  window.fetch = function(input, init) {
    let url = typeof input === 'string' ? input : (input instanceof Request ? input.url : String(input));

    const local = cdnToLocal(url);
    if (local) {
      if (typeof local === 'object' && local.isPackageJson) {
        // Return a fake package.json with the main file info
        const pkgName = local.pkg;
        const mainPath = findLocalPath(pkgName);
        if (mainPath) {
          const fakePackageJson = {
            name: pkgName.split('@')[0],
            version: pkgName.split('@')[1] || '1.0.0',
            main: mainPath.split('/').pop(),
            jsdelivr: mainPath.split('/').pop(),
            unpkg: mainPath.split('/').pop()
          };
          return Promise.resolve(new Response(JSON.stringify(fakePackageJson), {
            status: 200,
            headers: { 'Content-Type': 'application/json' }
          }));
        }
      } else {
        // Redirect to local file
        url = local;
        if (typeof input === 'string') {
          input = url;
        } else if (input instanceof Request) {
          input = new Request(url, input);
        }
      }
    }

    return originalFetch.call(this, input, init);
  };

  // Also intercept dynamic script loading
  const originalCreateElement = document.createElement.bind(document);
  document.createElement = function(tagName) {
    const element = originalCreateElement(tagName);
    if (tagName.toLowerCase() === 'script') {
      const originalSetAttribute = element.setAttribute.bind(element);
      element.setAttribute = function(name, value) {
        if (name === 'src') {
          const local = cdnToLocal(value);
          if (local && typeof local === 'string') {
            value = local;
          }
        }
        return originalSetAttribute(name, value);
      };

      // Also override the src property
      Object.defineProperty(element, 'src', {
        set: function(value) {
          const local = cdnToLocal(value);
          if (local && typeof local === 'string') {
            value = local;
          }
          originalSetAttribute('src', value);
        },
        get: function() {
          return element.getAttribute('src');
        }
      });
    }
    return element;
  };

  console.log('[OJS Offline] Interceptor loaded with', Object.keys(depMap).length, 'dependency mappings');
})();
</script>
]=]

  -- Combine the script parts with the dependency map
  local script = script_part1 .. dep_map_json .. script_part2

  -- Inject the script in the header (runs before quarto-ojs-runtime.js)
  quarto.doc.include_text("in-header", script)

  -- Copy the libs directory to the output using add_html_dependency
  -- The files will be available at the paths specified in dependency-map.json
  quarto.doc.add_html_dependency({
    name = "ojs-offline-libs",
    version = "1.0.0",
    resources = {{ path = "resources/libs", destination = "ojs-offline-libs" }}
  })

  return meta
end

-- Return the filter
return {
  { Meta = Meta }
}
