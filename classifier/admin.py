from django.contrib import admin
from .models import ProblemStatement, ModelContribution
from django.utils.html import format_html
from django.urls import reverse
from django.utils import timezone

# Register your models here.

@admin.register(ProblemStatement)
class ProblemStatementAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'model_type', 'selected_model', 'created_at']
    list_filter = ['model_type', 'created_at']
    search_fields = ['title', 'description', 'selected_model']
    readonly_fields = ['created_at', 'updated_at']
    
    fieldsets = (
        ('Basic Information', {
            'fields': ('title', 'description', 'model_type')
        }),
        ('Model Files', {
            'fields': ('model_file', 'scaler_file', 'selected_model')
        }),
        ('Data & Features', {
            'fields': ('dataset_sample', 'features_description', 'accuracy_scores')
        }),
        ('Code', {
            'fields': ('code_snippet', 'model_info')
        }),
        ('Learning Content', {
            'fields': ('problem_statement_detail', 'approach_explanation', 
                      'preprocessing_steps', 'model_architecture')
        }),
        ('Timestamps', {
            'fields': ('created_at', 'updated_at'),
            'classes': ('collapse',)
        }),
    )


@admin.register(ModelContribution)
class ModelContributionAdmin(admin.ModelAdmin):
    list_display = ['id', 'title', 'contributor_link', 'model_type', 'status_badge', 'submitted_at', 'quick_actions']
    list_filter = ['status', 'model_type', 'submitted_at']
    search_fields = ['title', 'description', 'contributor__username']
    readonly_fields = ['contributor', 'submitted_at', 'reviewed_at', 'updated_at', 'file_info']
    
    fieldsets = (
        ('Status & Review', {
            'fields': ('status', 'admin_notes', 'contributor', 'submitted_at', 'reviewed_at')
        }),
        ('Basic Information', {
            'fields': ('title', 'description', 'model_type', 'selected_model')
        }),
        ('Uploaded Files', {
            'fields': ('file_info', 'model_file', 'scaler_file', 'dataset_file', 'code_file')
        }),
        ('Data & Features', {
            'fields': ('dataset_sample', 'features_description', 'accuracy_scores')
        }),
        ('Code & Requirements', {
            'fields': ('code_snippet', 'requirements', 'model_info')
        }),
        ('Learning Content', {
            'fields': ('problem_statement_detail', 'approach_explanation', 
                      'preprocessing_steps', 'model_architecture')
        }),
    )
    
    actions = ['approve_contributions', 'reject_contributions']
    
    def contributor_link(self, obj):
        return format_html(
            '<a href="/admin/auth/user/{}/change/">{}</a>',
            obj.contributor.id,
            obj.contributor.username
        )
    contributor_link.short_description = 'Contributor'
    
    def status_badge(self, obj):
        colors = {
            'pending': '#ffc107',
            'approved': '#28a745',
            'rejected': '#dc3545'
        }
        return format_html(
            '<span style="background-color: {}; color: white; padding: 3px 10px; border-radius: 3px;">{}</span>',
            colors.get(obj.status, '#6c757d'),
            obj.get_status_display()
        )
    status_badge.short_description = 'Status'
    
    def quick_actions(self, obj):
        if obj.status == 'pending':
            return format_html(
                '<a class="button" href="{}">Approve</a> '
                '<a class="button" href="{}">Reject</a>',
                reverse('approve_contribution', args=[obj.id]),
                reverse('reject_contribution', args=[obj.id])
            )
        return '-'
    quick_actions.short_description = 'Actions'
    
    def file_info(self, obj):
        info = []
        if obj.model_file:
            info.append(f"Model: {obj.model_file.name}")
        if obj.scaler_file:
            info.append(f"Scaler: {obj.scaler_file.name}")
        if obj.dataset_file:
            info.append(f"Dataset: {obj.dataset_file.name}")
        if obj.code_file:
            info.append(f"Code: {obj.code_file.name}")
        return format_html('<br>'.join(info)) if info else 'No files uploaded'
    file_info.short_description = 'Uploaded Files'
    
    def approve_contributions(self, request, queryset):
        for contribution in queryset.filter(status='pending'):
            contribution.status = 'approved'
            contribution.reviewed_at = timezone.now()
            contribution.save()
            try:
                from .views import create_problem_statement_from_contribution
                create_problem_statement_from_contribution(contribution)
            except Exception as e:
                self.message_user(request, f"Error creating problem statement for {contribution.title}: {e}", level='ERROR')
        self.message_user(request, f"{queryset.count()} contributions approved!")
    approve_contributions.short_description = "Approve selected contributions"
    
    def reject_contributions(self, request, queryset):
        updated = queryset.update(status='rejected', reviewed_at=timezone.now())
        self.message_user(request, f"{updated} contributions rejected.")
    reject_contributions.short_description = "Reject selected contributions"
