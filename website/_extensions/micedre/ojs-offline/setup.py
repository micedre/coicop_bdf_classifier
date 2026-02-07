#!/usr/bin/env python3
"""
Download Observable JS dependencies for offline use

This script downloads all required JavaScript libraries for offline Observable JS
rendering in Quarto. Run this script once after installing the extension.

Usage:
    python3 setup.py

Custom libraries can be added in _quarto.yml:
    ojs-offline:
      libraries:
        - name: "moment"
          version: "2.29.4"
          files:
            - "min/moment.min.js"
"""

import argparse
import io
import json
import os
import re
import sys
import tarfile
import tempfile
import urllib.request
import urllib.error
import urllib.parse
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Dict, List, Any, Optional

# Try to import yaml, fall back to simple parsing if not available
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

# Dependencies to download
# Each dependency can have "files" (required) and "optional_files" (nice to have)
DEPENDENCIES = {
    # Core Observable libraries
    "@observablehq/inputs": {
        "version": "0.10.6",
        "files": [
            "dist/inputs.min.js"
        ],
        "optional_files": [
            "dist/inputs.min.js.map"
        ]
    },
    "@observablehq/plot": {
        "version": "0.6.11",
        "files": [
            "dist/plot.umd.min.js"
        ],
        "optional_files": [
            "dist/plot.umd.min.js.map"
        ]
    },
    "@observablehq/graphviz": {
        "version": "0.2.1",
        "files": [
            "dist/graphviz.min.js"
        ]
    },
    "@observablehq/highlight.js": {
        "version": "2.0.0",
        "files": [
            "highlight.min.js"
        ]
    },
    "@observablehq/katex": {
        "version": "0.11.1",
        "files": [
            "dist/katex.min.js",
            "dist/katex.min.css"
        ]
    },
    # Data visualization
    "d3": {
        "version": "7.8.5",
        "files": [
            "dist/d3.min.js"
        ],
        "optional_files": [
            "dist/d3.min.js.map"
        ]
    },
    "vega": {
        "version": "5.22.1",
        "files": [
            "build/vega.min.js"
        ],
        "optional_files": [
            "build/vega.min.js.map"
        ]
    },
    "vega-lite": {
        "version": "5.6.0",
        "files": [
            "build/vega-lite.min.js"
        ],
        "optional_files": [
            "build/vega-lite.min.js.map"
        ]
    },
    "vega-lite-api": {
        "version": "5.0.0",
        "files": [
            "build/vega-lite-api.min.js"
        ],
        "optional_files": [
            "build/vega-lite-api.min.js.map"
        ]
    },
    # Data processing
    "arquero": {
        "version": "4.8.8",
        "files": [
            "dist/arquero.min.js"
        ],
        "optional_files": [
            "dist/arquero.min.js.map"
        ]
    },
    "apache-arrow": {
        "version": "11.0.0",
        "files": [
            "Arrow.es2015.min.js"
        ],
        "optional_files": [
            "Arrow.es2015.min.js.map"
        ]
    },
    "@duckdb/duckdb-wasm": {
        "version": "1.24.0",
        "files": [
            "dist/duckdb-mvp.wasm",
            "dist/duckdb-eh.wasm",
            "dist/duckdb-browser-mvp.worker.js",
            "dist/duckdb-browser-eh.worker.js"
        ]
    },
    "sql.js": {
        "version": "1.8.0",
        "files": [
            "dist/sql-wasm.js",
            "dist/sql-wasm.wasm"
        ]
    },
    # Utilities
    "htl": {
        "version": "0.3.1",
        "files": [
            "dist/htl.min.js"
        ],
        "optional_files": [
            "dist/htl.min.js.map"
        ]
    },
    "lodash": {
        "version": "4.17.21",
        "files": [
            "lodash.min.js"
        ]
    },
    "jszip": {
        "version": "3.10.1",
        "files": [
            "dist/jszip.min.js"
        ]
    },
    "marked": {
        "version": "0.3.12",
        "files": [
            "marked.min.js"
        ]
    },
    "topojson-client": {
        "version": "3.1.0",
        "files": [
            "dist/topojson-client.min.js"
        ]
    },
    "exceljs": {
        "version": "4.3.0",
        "files": [
            "dist/exceljs.min.js"
        ]
    },
    "leaflet": {
        "version": "1.9.4",
        "files": [
            "dist/leaflet.js",
            "dist/leaflet.css"
        ]
    },
    "mermaid": {
        "version": "10.6.1",
        "files": [
            "dist/mermaid.min.js"
        ],
        "optional_files": [
            "dist/mermaid.min.js.map"
        ]
    }
}


class RegistryType(Enum):
    """Registry type for npm package downloads"""
    JSDELIVR = "jsdelivr"  # CDN-style direct file URLs
    NPM = "npm"            # Standard npm protocol (metadata + tarball)
    AUTO = "auto"          # Auto-detect based on URL


def detect_registry_type(registry_url: str) -> RegistryType:
    """
    Auto-detect the registry type based on URL hostname.

    CDN-style registries (jsdelivr, unpkg, cdnjs) serve extracted package files directly.
    Standard npm registries (npmjs.org, Nexus, Artifactory) use the npm protocol.
    """
    try:
        parsed = urllib.parse.urlparse(registry_url)
        hostname = parsed.netloc.lower()

        # CDN-style registries that serve files directly
        cdn_patterns = [
            'jsdelivr.net',
            'unpkg.com',
            'cdnjs.cloudflare.com',
            'cdn.skypack.dev',
            'esm.sh',
        ]

        for pattern in cdn_patterns:
            if pattern in hostname:
                return RegistryType.JSDELIVR

        # Everything else is assumed to be npm protocol
        return RegistryType.NPM
    except Exception:
        # Default to npm protocol on parse error
        return RegistryType.NPM


class RegistryStrategy(ABC):
    """Abstract base class for registry strategies"""

    def __init__(self, registry_url: str, timeout: int = 30):
        self.registry_url = registry_url.rstrip('/')
        self.timeout = timeout

    def _make_request(self, url: str, headers: Optional[Dict[str, str]] = None) -> bytes:
        """Make an HTTP request and return the response content"""
        req_headers = {'User-Agent': 'Quarto-OJS-Offline-Extension/1.0'}
        if headers:
            req_headers.update(headers)

        req = urllib.request.Request(url, headers=req_headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as response:
            return response.read()

    @abstractmethod
    def get_file(self, package_name: str, version: str, file_path: str) -> bytes:
        """Get file content from a package"""
        pass

    def cleanup(self):
        """Clean up any resources (override in subclasses if needed)"""
        pass


class JsDelivrRegistry(RegistryStrategy):
    """
    Registry strategy for CDN-style registries (jsdelivr, unpkg, etc.)

    URL pattern: {registry}/{package}@{version}/{file_path}
    """

    def get_file(self, package_name: str, version: str, file_path: str) -> bytes:
        """Get file directly from CDN URL"""
        url = f"{self.registry_url}/{package_name}@{version}/{file_path}"
        return self._make_request(url)


class TarballCache:
    """Cache for downloaded tarballs to avoid re-downloading"""

    def __init__(self):
        self._cache: Dict[str, tarfile.TarFile] = {}
        self._temp_files: List[tempfile.SpooledTemporaryFile] = []

    def get_or_download(self, key: str, download_func) -> tarfile.TarFile:
        """Get tarball from cache or download it"""
        if key not in self._cache:
            tarball_bytes = download_func()
            # Create a temporary file-like object
            temp_file = io.BytesIO(tarball_bytes)
            self._cache[key] = tarfile.open(fileobj=temp_file, mode='r:gz')
        return self._cache[key]

    def cleanup(self):
        """Close all cached tarballs"""
        for tf in self._cache.values():
            try:
                tf.close()
            except Exception:
                pass
        self._cache.clear()


class NpmRegistry(RegistryStrategy):
    """
    Registry strategy for standard npm registries (npmjs.org, Nexus, Artifactory)

    Protocol:
    1. Fetch package metadata from {registry}/{encoded_name}/{version}
    2. Get tarball URL from metadata.dist.tarball
    3. Download and extract the tarball
    4. Get files from inside package/ directory within the tarball
    """

    def __init__(self, registry_url: str, timeout: int = 30):
        super().__init__(registry_url, timeout)
        self._metadata_cache: Dict[str, Dict] = {}
        self._tarball_cache = TarballCache()

    def _encode_package_name(self, package_name: str) -> str:
        """
        Encode package name for npm registry URL.

        Scoped packages like @duckdb/duckdb-wasm need special handling:
        @duckdb/duckdb-wasm -> @duckdb%2Fduckdb-wasm
        """
        if package_name.startswith('@') and '/' in package_name:
            # Scoped package: encode the slash
            scope, name = package_name.split('/', 1)
            return f"{scope}%2F{name}"
        return package_name

    def _fetch_metadata(self, package_name: str, version: str) -> Dict:
        """Fetch package metadata from registry"""
        cache_key = f"{package_name}@{version}"

        if cache_key in self._metadata_cache:
            return self._metadata_cache[cache_key]

        encoded_name = self._encode_package_name(package_name)
        url = f"{self.registry_url}/{encoded_name}/{version}"

        try:
            content = self._make_request(url, headers={'Accept': 'application/json'})
            metadata = json.loads(content.decode('utf-8'))
            self._metadata_cache[cache_key] = metadata
            return metadata
        except Exception as e:
            raise RuntimeError(f"Failed to fetch metadata for {package_name}@{version}: {e}")

    def _get_tarball_url(self, package_name: str, version: str) -> str:
        """Get tarball URL from package metadata"""
        metadata = self._fetch_metadata(package_name, version)

        # Try to get tarball URL from metadata
        if 'dist' in metadata and 'tarball' in metadata['dist']:
            return metadata['dist']['tarball']

        # Fallback: construct URL
        encoded_name = self._encode_package_name(package_name)
        # Standard npm tarball URL pattern
        if package_name.startswith('@'):
            # Scoped package: @scope/name -> @scope/name/-/name-version.tgz
            scope, name = package_name.split('/', 1)
            return f"{self.registry_url}/{encoded_name}/-/{name}-{version}.tgz"
        else:
            return f"{self.registry_url}/{encoded_name}/-/{package_name}-{version}.tgz"

    def _download_tarball(self, package_name: str, version: str) -> tarfile.TarFile:
        """Download and cache tarball for a package"""
        cache_key = f"{package_name}@{version}"

        def download():
            tarball_url = self._get_tarball_url(package_name, version)
            return self._make_request(tarball_url)

        return self._tarball_cache.get_or_download(cache_key, download)

    def get_file(self, package_name: str, version: str, file_path: str) -> bytes:
        """Extract file from package tarball"""
        tf = self._download_tarball(package_name, version)

        # npm tarballs contain files in a 'package/' directory
        # Try with package/ prefix first, then without
        possible_paths = [
            f"package/{file_path}",
            file_path,
        ]

        for path in possible_paths:
            try:
                member = tf.getmember(path)
                file_obj = tf.extractfile(member)
                if file_obj:
                    return file_obj.read()
            except KeyError:
                continue

        # List available files for debugging
        available = [m.name for m in tf.getmembers() if m.isfile()][:10]
        raise FileNotFoundError(
            f"File '{file_path}' not found in tarball for {package_name}@{version}. "
            f"Available files (first 10): {available}"
        )

    def cleanup(self):
        """Clean up tarball cache"""
        self._tarball_cache.cleanup()


def create_registry(registry_url: str, registry_type: RegistryType, timeout: int = 30) -> RegistryStrategy:
    """Factory function to create the appropriate registry strategy"""
    if registry_type == RegistryType.AUTO:
        registry_type = detect_registry_type(registry_url)
        print(f"Auto-detected registry type: {registry_type.value}")

    if registry_type == RegistryType.JSDELIVR:
        return JsDelivrRegistry(registry_url, timeout)
    else:
        return NpmRegistry(registry_url, timeout)


def parse_yaml_simple(content: str) -> Dict[str, Any]:
    """
    Simple YAML parser for _quarto.yml - handles basic nested structures.
    Only parses what we need for ojs-offline configuration.
    """
    result = {}
    lines = content.split('\n')
    current_section = None
    current_lib = None
    current_list_key = None
    indent_stack = [(0, result)]

    for line in lines:
        # Skip empty lines and comments
        stripped = line.strip()
        if not stripped or stripped.startswith('#'):
            continue

        # Calculate indentation
        indent = len(line) - len(line.lstrip())

        # Handle list items
        if stripped.startswith('- '):
            item_content = stripped[2:].strip()
            if ':' in item_content:
                # It's a dict item like "- name: value"
                key, value = item_content.split(':', 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if current_list_key and isinstance(indent_stack[-1][1], list):
                    if not indent_stack[-1][1] or not isinstance(indent_stack[-1][1][-1], dict):
                        indent_stack[-1][1].append({})
                    indent_stack[-1][1][-1][key] = value
            else:
                # Simple list item
                item = item_content.strip('"').strip("'")
                if isinstance(indent_stack[-1][1], list):
                    indent_stack[-1][1].append(item)
            continue

        # Handle key: value pairs
        if ':' in stripped:
            key, value = stripped.split(':', 1)
            key = key.strip()
            value = value.strip()

            # Pop stack to correct level
            while len(indent_stack) > 1 and indent <= indent_stack[-1][0]:
                indent_stack.pop()

            current_dict = indent_stack[-1][1]
            if not isinstance(current_dict, dict):
                continue

            if value:
                # Simple key: value
                current_dict[key] = value.strip('"').strip("'")
            else:
                # Could be start of a list or nested dict
                # Check next line to determine
                current_dict[key] = {}
                indent_stack.append((indent + 2, current_dict[key]))
                current_list_key = key

    return result


def load_custom_libraries(quarto_yml_path: Path) -> Dict[str, Dict]:
    """
    Load additional libraries from _quarto.yml

    Expected format:
        ojs-offline:
          libraries:
            - name: "package-name"
              version: "1.0.0"
              files:
                - "dist/file.min.js"
              optional_files:
                - "dist/file.min.js.map"
    """
    if not quarto_yml_path.exists():
        return {}

    try:
        with open(quarto_yml_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if HAS_YAML:
            config = yaml.safe_load(content)
        else:
            # Use simple parser as fallback
            config = parse_yaml_simple(content)

        if not config:
            return {}

        ojs_config = config.get('ojs-offline', {})
        if not isinstance(ojs_config, dict):
            return {}

        libraries = ojs_config.get('libraries', [])
        if not isinstance(libraries, list):
            return {}

        result = {}
        for lib in libraries:
            if not isinstance(lib, dict):
                continue

            name = lib.get('name')
            version = lib.get('version')
            files = lib.get('files', [])

            if not name or not version or not files:
                print(f"⚠ Skipping invalid library entry: {lib}")
                continue

            result[name] = {
                'version': version,
                'files': files if isinstance(files, list) else [files],
                'optional_files': lib.get('optional_files', [])
            }

        if result:
            print(f"📋 Found {len(result)} custom libraries in _quarto.yml")

        return result

    except Exception as e:
        print(f"⚠ Warning: Could not parse _quarto.yml: {e}")
        return {}


def load_registry_config(config_file: Path = None) -> Dict:
    """Load registry configuration from file"""
    if config_file and config_file.exists():
        with open(config_file) as f:
            return json.load(f)

    # Try default config file location
    default_config = Path(__file__).parent / "registry-config.json"
    if default_config.exists():
        with open(default_config) as f:
            return json.load(f)

    return {}


def validate_registry_url(url: str) -> bool:
    """Validate that the registry URL is properly formatted"""
    try:
        parsed = urllib.parse.urlparse(url)
        if not parsed.scheme in ['http', 'https']:
            print(f"⚠ Warning: Registry URL should use http or https")
            return False
        if not parsed.netloc:
            print(f"⚠ Warning: Invalid registry URL: {url}")
            return False
        return True
    except Exception as e:
        print(f"⚠ Warning: Could not parse registry URL: {e}")
        return False


class DependencyDownloader:
    def __init__(self, base_dir: Path, registry_url: str = None, timeout: int = 30,
                 dependencies: Dict = None, registry_type: RegistryType = RegistryType.AUTO):
        self.base_dir = base_dir
        self.libs_dir = base_dir / "resources" / "libs"
        self.dependency_map = {}
        self.failed_downloads = []
        self.failed_optional = []

        # Use provided dependencies or default
        self.dependencies = dependencies or DEPENDENCIES

        # Use custom registry or default
        self.registry_url = registry_url or "https://cdn.jsdelivr.net/npm"
        self.timeout = timeout

        # Validate registry URL
        if not validate_registry_url(self.registry_url):
            print(f"⚠ Warning: Proceeding with potentially invalid registry URL")

        # Create registry strategy
        self.registry = create_registry(self.registry_url, registry_type, timeout)


    def download_file(self, name: str, version: str, file_path: str,
                       local_path: Path, optional: bool = False) -> bool:
        """Download a file using the registry strategy"""
        try:
            print(f"  Downloading: {name}@{version}/{file_path}")
            local_path.parent.mkdir(parents=True, exist_ok=True)

            # Use registry strategy to get file content
            content = self.registry.get_file(name, version, file_path)

            with open(local_path, 'wb') as f:
                f.write(content)

            file_size = len(content) / 1024  # KB
            print(f"    ✓ Saved to: {local_path.relative_to(self.base_dir)} ({file_size:.1f} KB)")
            return True

        except urllib.error.HTTPError as e:
            if optional:
                print(f"    ⊘ Optional file not found (skipping)")
                self.failed_optional.append((f"{name}@{version}/{file_path}", str(e)))
            else:
                print(f"    ✗ HTTP Error {e.code}: {e.reason}")
                self.failed_downloads.append((f"{name}@{version}/{file_path}", str(e)))
            return False
        except urllib.error.URLError as e:
            if optional:
                print(f"    ⊘ Optional file unavailable (skipping)")
                self.failed_optional.append((f"{name}@{version}/{file_path}", str(e)))
            else:
                print(f"    ✗ URL Error: {e.reason}")
                self.failed_downloads.append((f"{name}@{version}/{file_path}", str(e)))
            return False
        except FileNotFoundError as e:
            if optional:
                print(f"    ⊘ Optional file not found in tarball (skipping)")
                self.failed_optional.append((f"{name}@{version}/{file_path}", str(e)))
            else:
                print(f"    ✗ File not found: {e}")
                self.failed_downloads.append((f"{name}@{version}/{file_path}", str(e)))
            return False
        except Exception as e:
            if optional:
                print(f"    ⊘ Optional file error (skipping)")
                self.failed_optional.append((f"{name}@{version}/{file_path}", str(e)))
            else:
                print(f"    ✗ Error: {e}")
                self.failed_downloads.append((f"{name}@{version}/{file_path}", str(e)))
            return False

    def download_package(self, name: str, config: Dict) -> None:
        """Download all files for a package"""
        version = config["version"]
        files = config.get("files", [])
        optional_files = config.get("optional_files", [])

        print(f"\n📦 {name}@{version}")

        # Download required files
        for file_path in files:
            # Construct local path
            local_path = self.libs_dir / f"{name}@{version}" / file_path

            # Download file using registry strategy
            if self.download_file(name, version, file_path, local_path, optional=False):
                # Add to dependency map
                # Map both with and without the package version for flexibility
                map_key = f"{name}@{version}/{file_path}"
                # Make path relative to the HTML document's libs directory
                map_value = f"ojs-offline-libs/{name}@{version}/{file_path}"
                self.dependency_map[map_key] = map_value

                # Also add base package mapping (points to main file)
                if file_path == files[0]:  # First file is the main entry
                    self.dependency_map[f"{name}@{version}"] = map_value
                    self.dependency_map[name] = map_value

        # Download optional files (like source maps)
        for file_path in optional_files:
            local_path = self.libs_dir / f"{name}@{version}" / file_path

            if self.download_file(name, version, file_path, local_path, optional=True):
                # Add to dependency map if downloaded successfully
                map_key = f"{name}@{version}/{file_path}"
                map_value = f"ojs-offline-libs/{name}@{version}/{file_path}"
                self.dependency_map[map_key] = map_value

    def save_dependency_map(self) -> None:
        """Save dependency map to JSON file"""
        map_path = self.base_dir / "resources" / "dependency-map.json"
        print(f"\n💾 Saving dependency map to: {map_path.relative_to(self.base_dir)}")

        with open(map_path, 'w') as f:
            json.dump(self.dependency_map, f, indent=2, sort_keys=True)

        print(f"   ✓ Saved {len(self.dependency_map)} entries")

    def print_summary(self) -> None:
        """Print download summary"""
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)

        total_size = 0
        for root, dirs, files in os.walk(self.libs_dir):
            for file in files:
                file_path = Path(root) / file
                total_size += file_path.stat().st_size

        print(f"✓ Total packages: {len(self.dependencies)}")
        print(f"✓ Total files: {len(self.dependency_map) // 3}")  # Rough estimate
        print(f"✓ Total size: {total_size / (1024*1024):.1f} MB")

        if self.failed_downloads:
            print(f"\n⚠ FAILED REQUIRED DOWNLOADS: {len(self.failed_downloads)}")
            for url, error in self.failed_downloads:
                print(f"  - {url}")
                print(f"    Error: {error}")
        else:
            print("\n✓ All required files downloaded successfully!")

        if self.failed_optional:
            print(f"\nℹ Skipped optional files: {len(self.failed_optional)} (source maps)")
            print("  These are debug files - not needed for functionality")

    def run(self) -> int:
        """Run the download process"""
        print("="*60)
        print("Quarto OJS Offline - Dependency Downloader")
        print("="*60)
        print(f"Target directory: {self.base_dir}")
        print(f"Libraries directory: {self.libs_dir}")
        print(f"Registry: {self.registry_url}")
        print(f"Timeout: {self.timeout}s")
        print()

        try:
            # Create directories
            self.libs_dir.mkdir(parents=True, exist_ok=True)

            # Download all packages
            for name, config in self.dependencies.items():
                self.download_package(name, config)

            # Save dependency map
            self.save_dependency_map()

            # Print summary
            self.print_summary()

            # Return exit code
            return 1 if self.failed_downloads else 0
        finally:
            # Cleanup registry resources (e.g., cached tarballs)
            self.registry.cleanup()


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Download Observable JS dependencies for offline use"
    )
    parser.add_argument(
        '-r', '--registry',
        help='Custom npm registry URL (default: https://cdn.jsdelivr.net/npm)',
        default=None
    )
    parser.add_argument(
        '-c', '--config',
        type=Path,
        help='Path to registry configuration file',
        default=None
    )
    parser.add_argument(
        '--timeout',
        type=int,
        help='Download timeout in seconds (default: 30)',
        default=30
    )
    parser.add_argument(
        '-t', '--registry-type',
        choices=['jsdelivr', 'npm', 'auto'],
        help='Registry type: jsdelivr (CDN-style), npm (standard npm protocol), auto (default: auto-detect)',
        default=None
    )

    args = parser.parse_args()

    # Load configuration with priority: CLI > ENV > Config file > Default
    config = load_registry_config(args.config)

    registry_url = (
        args.registry or
        os.environ.get('NPM_REGISTRY') or
        config.get('registry') or
        "https://cdn.jsdelivr.net/npm"
    )

    timeout = (
        args.timeout if args.timeout != 30 else
        int(os.environ.get('NPM_TIMEOUT', config.get('timeout', 30)))
    )

    # Determine registry type with priority: CLI > ENV > Config file > Default (auto)
    registry_type_str = (
        getattr(args, 'registry_type', None) or
        os.environ.get('NPM_REGISTRY_TYPE') or
        config.get('registry_type') or
        'auto'
    )
    registry_type = RegistryType(registry_type_str)

    # Determine base directory (where this script is located)
    script_dir = Path(__file__).parent.resolve()

    # Find project root (parent of _extensions directory)
    # Script is at: project/_extensions/ojs-offline/setup.py
    project_root = script_dir.parent.parent
    quarto_yml = project_root / "_quarto.yml"

    # Load custom libraries from _quarto.yml
    custom_libs = load_custom_libraries(quarto_yml)

    # Merge: custom libraries are added to built-in ones
    # Custom libs with same name will override built-in
    all_dependencies = {**DEPENDENCIES, **custom_libs}

    # Run downloader with merged dependencies
    downloader = DependencyDownloader(
        script_dir, registry_url, timeout,
        dependencies=all_dependencies,
        registry_type=registry_type
    )
    exit_code = downloader.run()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
