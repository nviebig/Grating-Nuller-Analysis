## GitHub Copilot Chat

- Extension Version: 0.22.4 (prod)
- VS Code: vscode/1.95.3
- OS: Mac

## Network

User Settings:
```json
  "github.copilot.advanced": {
    "debug.useElectronFetcher": true,
    "debug.useNodeFetcher": false
  }
```

Connecting to https://api.github.com:
- DNS ipv4 Lookup: 140.82.121.6 (6 ms)
- DNS ipv6 Lookup: ::ffff:140.82.121.6 (2 ms)
- Electron Fetcher (configured): HTTP 200 (39 ms)
- Node Fetcher: HTTP 200 (38 ms)
- Helix Fetcher: HTTP 200 (107 ms)

Connecting to https://api.individual.githubcopilot.com/_ping:
- DNS ipv4 Lookup: 140.82.113.21 (54 ms)
- DNS ipv6 Lookup: ::ffff:140.82.113.21 (2 ms)
- Electron Fetcher (configured): HTTP 200 (294 ms)
- Node Fetcher: HTTP 200 (299 ms)
- Helix Fetcher: HTTP 200 (311 ms)

## Documentation

In corporate networks: [Troubleshooting firewall settings for GitHub Copilot](https://docs.github.com/en/copilot/troubleshooting-github-copilot/troubleshooting-firewall-settings-for-github-copilot).