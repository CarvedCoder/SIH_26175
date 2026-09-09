/** @module api/validation */
import { apiFetch } from './client.js';

/** GET /scenes/{id}/validation */
export const getValidation = (id) => apiFetch(`/scenes/${id}/validation`);

/** GET /scenes/{id}/validation/error-map */
export const getErrorMap = (id) => apiFetch(`/scenes/${id}/validation/error-map`);
