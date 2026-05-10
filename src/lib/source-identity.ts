import { getFileName, getRelativePath, normalizePath } from "@/lib/path-utils"

const RAW_SOURCES_PREFIX = "raw/sources/"
const RAW_VECTOR_PREFIX = "raw-sources/"
const WIKI_PREFIX = "wiki/"

function stripTrailingSlash(path: string): string {
  return path.replace(/\/+$/, "")
}

function stripLastExtension(path: string): string {
  const slash = path.lastIndexOf("/")
  const dot = path.lastIndexOf(".")
  if (dot > slash) return path.slice(0, dot)
  return path
}

function stripRawSourcesPrefix(path: string): string {
  const normalized = normalizePath(path).replace(/^\/+/, "")
  if (normalized.startsWith(RAW_SOURCES_PREFIX)) {
    return normalized.slice(RAW_SOURCES_PREFIX.length)
  }
  const marker = `/${RAW_SOURCES_PREFIX}`
  const idx = normalized.indexOf(marker)
  if (idx >= 0) return normalized.slice(idx + marker.length)
  return normalized
}

export function projectRelativePath(projectPath: string, path: string): string {
  const pp = stripTrailingSlash(normalizePath(projectPath))
  const normalized = normalizePath(path)
  return getRelativePath(normalized, pp).replace(/^\/+/, "")
}

export function sourceRefForPath(projectPath: string, sourcePath: string): string {
  const rel = projectRelativePath(projectPath, sourcePath)
  if (rel.startsWith(RAW_SOURCES_PREFIX)) return rel
  const nested = stripRawSourcesPrefix(sourcePath)
  if (nested !== normalizePath(sourcePath).replace(/^\/+/, "")) {
    return `${RAW_SOURCES_PREFIX}${nested}`
  }
  return `${RAW_SOURCES_PREFIX}${getFileName(sourcePath)}`
}

export function sourceStemPathFromRef(sourceRef: string): string {
  return stripLastExtension(stripRawSourcesPrefix(sourceRef))
}

export function sourceSummaryPathForRef(sourceRef: string): string {
  return `wiki/sources/${sourceStemPathFromRef(sourceRef)}.md`
}

export function mediaDirForSourceRef(projectPath: string, sourceRef: string): string {
  return `${stripTrailingSlash(normalizePath(projectPath))}/wiki/media/${sourceStemPathFromRef(sourceRef)}`
}

export function rawSourceVectorId(sourceRef: string): string {
  return `${RAW_VECTOR_PREFIX}${stripRawSourcesPrefix(sourceRef)}`
}

export function sourceRefFromRawVectorId(id: string): string | null {
  if (!id.startsWith(RAW_VECTOR_PREFIX)) return null
  return `${RAW_SOURCES_PREFIX}${id.slice(RAW_VECTOR_PREFIX.length)}`
}

export function isRawSourceVectorId(id: string): boolean {
  return sourceRefFromRawVectorId(id) !== null
}

export function wikiPageIdForRelativePath(relativePath: string): string {
  let rel = normalizePath(relativePath).replace(/^\/+/, "")
  if (rel.startsWith(WIKI_PREFIX)) rel = rel.slice(WIKI_PREFIX.length)
  return stripLastExtension(rel)
}

export function wikiPageIdFromPath(projectPath: string, pagePath: string): string {
  const rel = projectRelativePath(projectPath, pagePath)
  if (rel.startsWith(WIKI_PREFIX) || !normalizePath(pagePath).includes("/wiki/")) {
    return wikiPageIdForRelativePath(rel)
  }
  return wikiPageIdForRelativePath(normalizePath(pagePath).split("/wiki/").pop() ?? rel)
}

export function wikiPathForPageId(projectPath: string, pageId: string): string {
  return `${stripTrailingSlash(normalizePath(projectPath))}/wiki/${pageId}.md`
}

export function pageIdBasename(pageId: string): string {
  return getFileName(pageId)
}
