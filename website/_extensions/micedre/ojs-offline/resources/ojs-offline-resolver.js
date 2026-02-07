// ojs-offline-resolver.js
// Custom resolver for offline Observable JS dependencies

(function() {
  'use strict';

  // Configuration
  const DEBUG = false;
  const FALLBACK_TO_CDN = false;
  const CDN_BASE = 'https://cdn.jsdelivr.net/npm/';

  // Dependency mapping loaded from dependency-map.json
  let dependencyMap = {};

  // Log helper
  function log(...args) {
    if (DEBUG) {
      console.log('[OJS Offline]', ...args);
    }
  }

  // Load dependency map from embedded JSON
  async function loadDependencyMap() {
    try {
      // Try to load from embedded script tag
      const mapScript = document.getElementById('ojs-dependency-map');
      if (mapScript) {
        const mapUrl = mapScript.getAttribute('src');
        if (mapUrl) {
          const response = await fetch(mapUrl);
          dependencyMap = await response.json();
          log('Dependency map loaded:', Object.keys(dependencyMap).length, 'entries');
          return;
        }
      }

      // Fallback: try to fetch directly
      const response = await fetch('ojs-offline-dependency-map.json');
      if (response.ok) {
        dependencyMap = await response.json();
        log('Dependency map loaded from direct fetch');
      }
    } catch (error) {
      console.warn('[OJS Offline] Failed to load dependency map:', error);
    }
  }

  // Parse package identifier (e.g., "d3@7.8.5/dist/d3.min.js" or "@observablehq/plot@0.6.11")
  function parsePackageIdentifier(specifier) {
    // Handle scoped packages (@org/package@version/path)
    const scopedMatch = specifier.match(/^(@[^/]+\/[^@]+)@([^/]+)(?:\/(.+))?$/);
    if (scopedMatch) {
      return {
        name: scopedMatch[1],
        version: scopedMatch[2],
        path: scopedMatch[3] || ''
      };
    }

    // Handle regular packages (package@version/path)
    const regularMatch = specifier.match(/^([^@]+)@([^/]+)(?:\/(.+))?$/);
    if (regularMatch) {
      return {
        name: regularMatch[1],
        version: regularMatch[2],
        path: regularMatch[3] || ''
      };
    }

    // Handle bare package names (try to match without version)
    return {
      name: specifier,
      version: null,
      path: ''
    };
  }

  // Find local path for a package
  function findLocalPath(specifier) {
    // Direct lookup
    if (dependencyMap[specifier]) {
      log('Direct match:', specifier, '->', dependencyMap[specifier]);
      return dependencyMap[specifier];
    }

    // Try to parse and find closest match
    const parsed = parsePackageIdentifier(specifier);

    // Try with full path
    const fullKey = parsed.version
      ? `${parsed.name}@${parsed.version}${parsed.path ? '/' + parsed.path : ''}`
      : specifier;

    if (dependencyMap[fullKey]) {
      log('Parsed match:', fullKey, '->', dependencyMap[fullKey]);
      return dependencyMap[fullKey];
    }

    // Try package base (name@version)
    const baseKey = parsed.version ? `${parsed.name}@${parsed.version}` : parsed.name;
    if (dependencyMap[baseKey]) {
      log('Base match:', baseKey, '->', dependencyMap[baseKey]);
      return dependencyMap[baseKey];
    }

    // Try to find any version of the package
    const namePrefix = parsed.name + '@';
    for (const key in dependencyMap) {
      if (key.startsWith(namePrefix)) {
        log('Prefix match:', key, '->', dependencyMap[key]);
        return dependencyMap[key];
      }
    }

    log('No match found for:', specifier);
    return null;
  }

  // Resolve module specifier to URL
  function resolveModuleSpecifier(specifier, base) {
    // Check if it's already a full URL
    if (specifier.startsWith('http://') || specifier.startsWith('https://')) {
      return specifier;
    }

    // Check if it's a relative path
    if (specifier.startsWith('./') || specifier.startsWith('../')) {
      return new URL(specifier, base || window.location.href).href;
    }

    // Try to find local path
    const localPath = findLocalPath(specifier);
    if (localPath) {
      // Construct relative path from the HTML document
      // Quarto typically places extensions resources in {doc}_files/libs/{extension}/
      const baseUrl = base || window.location.href;
      return new URL(localPath, baseUrl).href;
    }

    // Fallback to CDN if enabled
    if (FALLBACK_TO_CDN) {
      log('Falling back to CDN for:', specifier);
      return CDN_BASE + specifier;
    }

    // No resolution possible
    console.error('[OJS Offline] Cannot resolve module:', specifier);
    return null;
  }

  // Override the define function for AMD modules
  function setupAMDInterception() {
    if (typeof window.define === 'function' && window.define.amd) {
      const originalDefine = window.define;
      window.define = function(...args) {
        log('AMD define called with', args.length, 'arguments');
        return originalDefine.apply(this, args);
      };
      window.define.amd = originalDefine.amd;
    }
  }

  // Create a custom require function for OJS
  function createOfflineRequire() {
    return async function ojsOfflineRequire(specifier) {
      log('Require called for:', specifier);

      const resolvedUrl = resolveModuleSpecifier(specifier);
      if (!resolvedUrl) {
        throw new Error(`Cannot resolve module: ${specifier}`);
      }

      log('Resolved to:', resolvedUrl);

      // Try to load as ES module
      try {
        const module = await import(resolvedUrl);
        log('Loaded as ES module:', specifier);
        return module;
      } catch (esError) {
        log('ES module load failed, trying script injection:', esError.message);

        // Fallback: inject as script tag (for UMD modules)
        return new Promise((resolve, reject) => {
          const script = document.createElement('script');
          script.src = resolvedUrl;
          script.onload = () => {
            log('Script loaded:', specifier);
            // Try to find the module in common global locations
            const moduleName = specifier.split('@')[0].split('/').pop();
            const module = window[moduleName] || window[specifier] || {};
            resolve(module);
          };
          script.onerror = () => {
            console.error('[OJS Offline] Failed to load script:', resolvedUrl);
            reject(new Error(`Failed to load module: ${specifier}`));
          };
          document.head.appendChild(script);
        });
      }
    };
  }

  // Hook into Quarto's OJS runtime
  function hookIntoQuartoOJS() {
    // Wait for the OJS runtime to be available
    const checkInterval = setInterval(() => {
      // Check if Quarto has set up its module system
      if (window._ojs || window.mainOjs || typeof window.define === 'function') {
        clearInterval(checkInterval);
        log('Quarto OJS runtime detected, installing offline resolver');

        // Install our custom require function
        if (window._ojs && typeof window._ojs.require === 'function') {
          const originalRequire = window._ojs.require;
          window._ojs.require = async function(specifier) {
            const localPath = findLocalPath(specifier);
            if (localPath) {
              log('Intercepting require for:', specifier);
              const offlineRequire = createOfflineRequire();
              return offlineRequire(specifier);
            }
            return originalRequire(specifier);
          };
        }

        setupAMDInterception();
      }
    }, 50);

    // Stop checking after 10 seconds
    setTimeout(() => {
      clearInterval(checkInterval);
    }, 10000);
  }

  // Initialize on DOMContentLoaded
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', async () => {
      await loadDependencyMap();
      hookIntoQuartoOJS();
    });
  } else {
    loadDependencyMap().then(() => {
      hookIntoQuartoOJS();
    });
  }

  // Export for external access
  window.__quartoOjsOffline = {
    resolveModuleSpecifier,
    findLocalPath,
    getDependencyMap: () => dependencyMap,
    setDebug: (enabled) => { DEBUG = enabled; }
  };

  log('OJS Offline resolver initialized');
})();
