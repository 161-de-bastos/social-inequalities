#!/usr/bin/bash
set -euo pipefail

RESOURCE_GROUP="exp-run-$(date +%Y%m%d%H%M%S)"
LOCATION="brazilsouth"
IMAGE="canonical:ubuntu-24_04-lts:server:latest"
VM_SIZE="Standard_E16s_v3"
ADMIN_USER="azureuser"

BASE_DIR="$(pwd)"
APP="$BASE_DIR/app"
DF="$BASE_DIR/df.csv"
CONFIGS_DIR="$BASE_DIR/configs"
RESULTS_DIR="$BASE_DIR/results"

MAX_JOBS=16
running=0

mkdir -p "$RESULTS_DIR"

echo "Creando Resource Group efímero: $RESOURCE_GROUP ..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --only-show-errors >/dev/null
echo "Resource Group creado."
echo

mapfile -t EXPERIMENTS < <(
    find "$CONFIGS_DIR" -type f -name "*.yaml" -printf "%f\n" |
    sed 's/.yaml$//' | sort
)

echo "Experimentos detectados:"
printf '  - %s\n' "${EXPERIMENTS[@]}"
echo

run_exp() {
  local exp="$1"

  echo "[$exp] Creando VM..."

  az vm create \
    --resource-group "$RESOURCE_GROUP" \
    --name "$exp" \
    --image "$IMAGE" \
    --size "$VM_SIZE" \
    --location "$LOCATION" \
    --admin-username "$ADMIN_USER" \
    --generate-ssh-keys \
    --only-show-errors >/dev/null

  echo "[$exp] VM creada. Esperando a que esté lista..."

  az vm wait \
    --resource-group "$RESOURCE_GROUP" \
    --name "$exp" \
    --created >/dev/null

  ip=$(az vm show -d \
    --resource-group "$RESOURCE_GROUP" \
    --name "$exp" \
    --query publicIps -o tsv)

  echo "[$exp] IP = $ip"

  cfg="$CONFIGS_DIR/$exp.yaml"
  res_dir="$RESULTS_DIR/$exp"
  mkdir -p "$res_dir"

  echo "[$exp] Preparando directorio remoto y subiendo archivos..."
  ssh -o StrictHostKeyChecking=accept-new "$ADMIN_USER@$ip" "mkdir -p $exp"
  scp -o StrictHostKeyChecking=accept-new "$APP" "$ADMIN_USER@$ip:$exp/app"
  scp -o StrictHostKeyChecking=accept-new "$DF"  "$ADMIN_USER@$ip:$exp/df.csv"
  scp -o StrictHostKeyChecking=accept-new "$cfg" "$ADMIN_USER@$ip:$exp/config.yaml"

  echo "[$exp] Ejecutando app en la VM..."
  ssh -o StrictHostKeyChecking=accept-new "$ADMIN_USER@$ip" "sudo apt-get update -y && sudo apt-get install -y build-essential libgomp1 libstdc++6 libc6 ca-certificates && cd $exp && chmod +x app && ./app"

  echo "[$exp] Descargando resultados..."
  scp -o StrictHostKeyChecking=accept-new "$ADMIN_USER@$ip:$exp/clusters.csv" "$res_dir/" 2>/dev/null || echo "[$exp] clusters.csv no encontrado"
  scp -o StrictHostKeyChecking=accept-new "$ADMIN_USER@$ip:$exp/results.log"  "$res_dir/" 2>/dev/null || echo "[$exp] r.log no encontrado"

  echo "[$exp] Borrando VM..."
  az vm delete \
    --resource-group "$RESOURCE_GROUP" \
    --name "$exp" \
    --yes \
    --only-show-errors >/dev/null

  echo "[$exp] COMPLETADO (VM eliminada)."
}

echo "=== Ejecutando VMs (máx $MAX_JOBS simultáneos) ==="

for exp in "${EXPERIMENTS[@]}"; do
  run_exp "$exp" &
  ((running++))
  
  if (( running >= MAX_JOBS )); then
    wait -n
    ((running--))
  fi
done

wait
echo
echo "========================================="
echo "  TODO TERMINÓ. Resultados en ./results/"
echo "  No quedan VMs vivas para estos runs."
echo "========================================="