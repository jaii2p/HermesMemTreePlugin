# Deploying the memtree MCP connector on an AWS Linux box

Goal: a public HTTPS URL for the Streamable HTTP endpoint, so Claude (claude.ai /
mobile / desktop, connected from *Anthropic's cloud*) can talk to his tree.

## Stack
    client (claude.ai connector, https://...)   -->   apache2 :443 (TLS)   -->   memtree http :8080 (127.0.0.1)

## 1. Ship the package
    scp -r /path/to/memtree  ubuntu@<aws-public-ip>:~/memtree
    ssh ubuntu@<aws-public-ip> 'sudo apt update && sudo apt install -y python3-venv apache2 certbot python3-certbot-apache'

## 2. Run the http server (its own tree in ./var)
    cd ~/memtree
    python3 -m venv .venv && .venv/bin/pip install numpy
    .venv/bin/python -m memtree_mcp.http --host 127.0.0.1 --port 8080
      # or:   ./run_http.sh --port 8080
    # keep it up as a service (systemd -> sudo tee /etc/systemd/system/memtree.service; see below)

## 3. Apache reverse proxy + Let's Encrypt
    sudo cp deploy/apache.conf /etc/apache2/sites-available/memtree.conf
    sudo sed -i 's/SERVERNAME/your-domain/g' /etc/apache2/sites-available/memtree.conf
    sudo a2enmod proxy proxy_http ssl headers
    sudo a2ensite memtree
    sudo certbot --apache -d your-domain
    sudo systemctl reload apache2
    # point your DNS A record (your-domain) at the box's public IP first

## 4. Give the URL to Claude
    claude.ai > Customize > Connectors > Add custom connector
      Web  ->  https://your-domain/mcp
    Free plan = 1 connector. No API key, no tokens.

## AWS security group
    Inbound: 443/tcp from 0.0.0.0/0 (the connector must be reachable), 22 from your IP.
    The http listener is 127.0.0.1 only; the firewall never sees :8080.

## systemd unit (run_http as a service)
    [Unit]
    Description=memtree MCP (streamable http)
    After=network.target
    [Service]
    User=ubuntu
    WorkingDirectory=/home/ubuntu/memtree
    Environment=PYTHONPATH=/home/ubuntu/memtree
    Environment=MEMTREE_HOME=/home/ubuntu/memtree/var
    ExecStart=/home/ubuntu/memtree/.venv/bin/python -m memtree_mcp.http --host 127.0.0.1 --port 8080
    Restart=always
    [Install]
    WantedBy=multi-user.target

## Health
    curl https://your-domain/healthz   # -> {"ok": true, "nodes": ...}

## Read/write posture
    A connector is open by design: anyone with the URL can call remember/sync.
    For a personal Free connector that is usually fine. If you want it locked down:
      - require a token: run the server with MEMTREE_TOKEN set and have Claude send it as
        `Authorization: Bearer <token>` (Claude Code -H; claude.ai OAuth/static headers vary
        by plan), or
      - keep writes read-only via a `read_only` home and only open remember to trusted callers.
    The mailbox bridge can reuse this same /mcp with a second peer -- each person
    gets their own branch and own whispers.
