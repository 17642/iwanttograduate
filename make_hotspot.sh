#!/bin/bash

WLAN=wlan0
SSID="RPI_HOTSPOT_319"
PASSPHRASE="RPIHOTSPOT319"

HOTSPOT_IP = 10.42.0.1
SUBNET=10.42.0
NETMASK=255.255.255.0

HOSTNAME=${hostname}

echo "[1/7] Stopping nmcli wifi..."
echo "If you want to use wifi with nmcli, use 'nmcli radio wifi on'."
nmcli radio wifi off
sudo rfkill unblock all

echo "[2/7] Stopping Services..."
sudo systemctl stop hostapd
sudo systemctl stop dnsmasq

echo "[3/7] Setting up static IP on $WLAN..."
sudo ip link set $WLAN down
sudo ip addr flush dev $WLAN
sudo ip addr add 10.42.0.1/24 dev $WLAN
sudo ip link set $WLAN up

echo "[4/7] Writing hostapd.conf ..."
cat <<EOF | sudo tee /etc/hostapd/hostapd.conf > /dev/null
interface=$WLAN
driver=nl80211
ssid=$SSID
hw_mode=g
channel=6
wmm_enabled=0
macaddr_acl=0
auth_algs=1
ignore_broadcast_ssid=0
wpa=2
wpa_passphrase=$PASSPHRASE
wpa_key_mgmt=WPA-PSK
rsn_pairwise=CCMP
EOF

sudo sed -i 's|#DAEMON_CONF=.*|DAEMON_CONF="/etc/hostapd/hostapd.conf"|' /etc/default/hostapd

echo "[5/7] Writing dnsmasq.conf ..."
sudo mv /etc/dnsmasq.conf /etc/dnsmasq.conf.bak 2>/dev/null
cat <<EOF | sudo tee /etc/dnsmasq.conf > /dev/null
interface=$WLAN
dhcp-range=${SUBNET}.10,${SUBNET}.100,12h
EOF

echo "[6/7] Enabling IP forwarding..."
sudo sed -i 's/#net.ipv4.ip_forward=1/net.ipv4.ip_forward=1/' /etc/sysctl.conf
sudo sysctl -p

echo "[7/7] Starting dnsmasq and hostapd..."
sudo systemctl restart dnsmasq
sudo systemctl unmask hostapd
sudo systemctl enable hostapd
sudo systemctl restart hostapd
echo "[V] Hotspot $SSID is running on $WLAN ($HOTSPOT_IP)"