import { defineConfig } from 'vitepress'

export default defineConfig({
  title: 'BayesianGPU',
  description: 'GPU-accelerated Bayesian inference for browser and Rust',
  base: '/bayesiangpu-core/',
  
  themeConfig: {
    nav: [
      { text: 'Guide', link: '/guide/' },
      { text: 'API', link: '/api/' },
      { text: 'GitHub', link: 'https://github.com/mojavedataops/bayesiangpu-core' }
    ],
    
    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/guide/' },
            { text: 'Quick Start', link: '/guide/quickstart' },
            { text: 'Installation', link: '/guide/installation' }
          ]
        },
        {
          text: 'Core Concepts',
          items: [
            { text: 'Model Definition', link: '/guide/models' },
            { text: 'Distributions', link: '/guide/distributions' },
            { text: 'Inference', link: '/guide/inference' },
            { text: 'Diagnostics', link: '/guide/diagnostics' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'API Reference',
          items: [
            { text: 'Overview', link: '/api/' },
            { text: 'Model', link: '/api/model' },
            { text: 'Distributions', link: '/api/distributions' },
            { text: 'Inference', link: '/api/inference' },
            { text: 'Results', link: '/api/results' }
          ]
        }
      ]
    },
    
    socialLinks: [
      { icon: 'github', link: 'https://github.com/mojavedataops/bayesiangpu-core' }
    ],
    
    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright 2026'
    }
  }
})
