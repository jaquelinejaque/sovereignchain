// Bundler for the Quorum VS Code extension.
// Bundles src/extension.ts -> dist/extension.js as a CJS Node18 module,
// with vscode marked external (the editor provides it at runtime).
//
// Flags:
//   --production  minify + drop sourcemaps
//   --watch       rebuild on file change

const esbuild = require('esbuild');

const production = process.argv.includes('--production');
const watch = process.argv.includes('--watch');

/** @type {import('esbuild').BuildOptions} */
const buildOptions = {
  entryPoints: ['src/extension.ts'],
  bundle: true,
  outfile: 'dist/extension.js',
  external: ['vscode'],
  platform: 'node',
  target: 'node18',
  format: 'cjs',
  sourcemap: !production,
  minify: production,
  logLevel: 'info'
};

async function run() {
  if (watch) {
    const ctx = await esbuild.context(buildOptions);
    await ctx.watch();
    console.log('[esbuild] watching for changes...');
  } else {
    await esbuild.build(buildOptions);
    console.log(`[esbuild] build complete (${production ? 'production' : 'development'})`);
  }
}

run().catch((err) => {
  console.error('[esbuild] build failed:', err);
  process.exit(1);
});
