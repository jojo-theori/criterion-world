#!/bin/zsh
# ccw.umjo.me 배포(옛 주소 cc.umjo.me 도 붙여 두고 index.html 이 새 주소로 넘긴다): git 에 올라간 web/ 파일만 모아 Cloudflare Workers 정적 자산으로 올린다.
# (로컬 전용 atlas *.js/*.jpg, strokes *.png 는 .gitignore 라 빠진다.) 커밋한 뒤 실행한다.
# 사용: scripts/deploy.sh
set -e
cd "$(dirname "$0")/.."
STAGE=$(mktemp -d)
git ls-files -z web | rsync -a --from0 --files-from=- ./ "$STAGE/"
cd "$STAGE"
npx wrangler deploy --assets ./web --name criterion-world --compatibility-date 2026-11-01 --domain ccw.umjo.me --domain cc.umjo.me
rm -rf "$STAGE"
